#include "ElemElemGraph.hpp"
#include "ElemElemGraphImpl.hpp"
#include "ElemGraphCoincidentElems.hpp"
#include "ElemGraphShellConnections.hpp"
#include "BulkDataIdMapper.hpp"

#include <vector>
#include <algorithm>

#include <stk_topology/topology.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Comm.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/baseImpl/DeletedElementInfo.hpp>
#include <stk_mesh/baseImpl/MeshImplUtils.hpp>

#include <stk_util/parallel/CommSparse.hpp>
#include <stk_util/environment/ReportHandler.hpp>
#include <stk_util/util/SortAndUnique.hpp>

namespace stk { namespace mesh {

impl::ElemSideToProcAndFaceId ElemElemGraph::get_element_side_ids_to_communicate() const
{
    stk::mesh::EntityVector elements_to_communicate;
    std::set<stk::mesh::Entity> element_set;
    const stk::mesh::BucketVector& shared_node_buckets = m_bulk_data.get_buckets(stk::topology::NODE_RANK, m_bulk_data.mesh_meta_data().globally_shared_part());
    for(size_t i=0; i<shared_node_buckets.size(); ++i)
    {
        const stk::mesh::Bucket& bucket = *shared_node_buckets[i];
        for(size_t node_index=0; node_index<bucket.size(); ++node_index)
        {
            stk::mesh::Entity node = bucket[node_index];
            const stk::mesh::Entity* elements = m_bulk_data.begin_elements(node);
            unsigned num_elements = m_bulk_data.num_elements(node);
            for(unsigned element_index=0; element_index<num_elements; ++element_index)
            {
                if (m_bulk_data.bucket(elements[element_index]).owned())
                {
                    element_set.insert(elements[element_index]);
                }
            }
        }
    }
    elements_to_communicate.assign(element_set.begin(), element_set.end());

    return impl::build_element_side_ids_to_proc_map(m_bulk_data, elements_to_communicate);
}

ElemElemGraph::ElemElemGraph(stk::mesh::BulkData& bulkData, const stk::mesh::Selector& sel, const stk::mesh::Selector* air) :
        m_bulk_data(bulkData),
        m_skinned_selector(sel),
        m_air_selector(air),
        m_parallelInfoForGraphEdges(bulkData.parallel_rank()),
        m_sideIdPool(bulkData)
{
    fill_from_mesh();
}

void ElemElemGraph::fill_from_mesh()
{
    clear_data_members();
    int numElems = size_data_members();

    impl::ElemSideToProcAndFaceId elem_side_comm;
    if (numElems > 0) {
        impl::fill_local_ids_and_fill_element_entities_and_topologies(m_bulk_data, m_local_id_to_element_entity, m_entity_to_local_id, m_element_topologies);
        fill_graph();

        elem_side_comm = get_element_side_ids_to_communicate();
    }

    // Todo: very conservative number (is it worth improving?)
    size_t num_side_ids_needed = elem_side_comm.size(); // parallel boundary faces
    num_side_ids_needed += m_graph.get_num_edges(); // locally_owned_faces

    num_side_ids_needed += 6*m_graph.get_num_elements_in_graph(); // skinned faces

    m_sideIdPool.generate_initial_ids(num_side_ids_needed);

    m_parallelInfoForGraphEdges.clear();
    fill_parallel_graph(elem_side_comm);

    extract_coincident_edges_and_fix_chosen_side_ids();

    GraphInfo graphInfo(m_graph, m_parallelInfoForGraphEdges, m_element_topologies);
    remove_graph_edges_blocked_by_shell(graphInfo);

    update_number_of_parallel_edges();
}

void ElemElemGraph::extract_coincident_edges_and_fix_chosen_side_ids()
{
    m_coincidentGraph = impl::extract_coincident_sides(m_graph, m_element_topologies);
    impl::BulkDataIdMapper idMapper(m_bulk_data, m_local_id_to_element_entity, m_entity_to_local_id);
    choose_face_id_for_coincident_elements(m_graph,
                                           m_parallelInfoForGraphEdges,
                                           m_coincidentGraph,
                                           idMapper,
                                           m_bulk_data.parallel());
}

void ElemElemGraph::extract_coincident_edges_and_fix_chosen_side_ids_for_specified_elems(const stk::mesh::EntityVector &elems)
{
    std::vector<impl::LocalId> localIds = get_local_ids_for_element_entities(elems);
    impl::append_extracted_coincident_sides(m_graph, m_element_topologies, localIds, m_coincidentGraph);
    impl::BulkDataIdMapper idMapper(m_bulk_data, m_local_id_to_element_entity, m_entity_to_local_id);
    choose_face_id_for_coincident_elements(m_graph,
                                           m_parallelInfoForGraphEdges,
                                           m_coincidentGraph,
                                           idMapper,
                                           m_bulk_data.parallel());
}

std::vector<impl::LocalId> ElemElemGraph::get_local_ids_for_element_entities(const stk::mesh::EntityVector &elems)
{
    std::vector<impl::LocalId> localIds;
    localIds.reserve(elems.size());
    for(stk::mesh::Entity elem : elems)
        localIds.push_back(m_entity_to_local_id[elem.local_offset()]);
    return localIds;
}

void ElemElemGraph::update_number_of_parallel_edges()
{
    // Figure out the real number of sides that can be generated, using
    // the rule of one side entity per element side (with multiple
    // coincident elements each being connected to the same side)
    m_num_parallel_edges = 0;
    for(size_t i = 0; i < m_graph.get_num_elements_in_graph(); ++i)
    {
        size_t numConnectedElems = m_graph.get_num_edges_for_element(i);
        for(size_t j = 0; j < numConnectedElems; ++j)
        {
            const GraphEdge & graphEdge = m_graph.get_edge_for_element(i, j);
            if(graphEdge.elem2 < 0)
            {
                m_num_parallel_edges++;
            }
        }
    }
}

ElemElemGraph::~ElemElemGraph() {}

size_t ElemElemGraph::get_num_connected_elems(stk::mesh::Entity local_element) const
{
    impl::LocalId local_id = get_local_element_id(local_element);
    return m_graph.get_num_edges_for_element(local_id);
}

unsigned ElemElemGraph::get_num_connected_elems(stk::mesh::Entity localElement, int side_id) const
{
    impl::LocalId local_id = get_local_element_id(localElement);
    std::vector<GraphEdge> graphEdges = m_graph.get_edges_for_element_side(local_id, side_id);
    return graphEdges.size();
}

bool ElemElemGraph::is_connected_elem_locally_owned(stk::mesh::Entity localElement, size_t indexConnElement) const
{
    impl::LocalId local_id = get_local_element_id(localElement);
    return m_graph.get_edge_for_element(local_id, indexConnElement).elem2 >= 0;
}

impl::ElementViaSidePair ElemElemGraph::get_connected_element_and_via_side(stk::mesh::Entity localElement, size_t indexConnElement) const
{
    impl::LocalId local_id = get_local_element_id(localElement);
    const GraphEdge & graphEdge = m_graph.get_edge_for_element(local_id, indexConnElement);
    impl::LocalId other_element_id = graphEdge.elem2;
    return {m_local_id_to_element_entity[other_element_id],graphEdge.side1};
}

impl::IdViaSidePair ElemElemGraph::get_connected_remote_id_and_via_side(stk::mesh::Entity localElement, size_t indexConnElement) const
{
    ThrowRequireMsg(!is_connected_elem_locally_owned(localElement, indexConnElement) , "Program error. Contact sierra-help@sandia.gov for support.");
    impl::LocalId local_id = get_local_element_id(localElement);
    const GraphEdge & graphEdge = m_graph.get_edge_for_element(local_id, indexConnElement);
    return {m_parallelInfoForGraphEdges.convert_negative_local_id_to_remote_global_id(graphEdge.elem2), graphEdge.side1};
}

int ElemElemGraph::get_connected_elements_side(stk::mesh::Entity localElement, size_t indexConnElement) const
{
    impl::LocalId local_id = get_local_element_id(localElement);
    const GraphEdge & graphEdge = m_graph.get_edge_for_element(local_id, indexConnElement);
    return graphEdge.side2;
}

int ElemElemGraph::get_owning_proc_id_of_remote_element(stk::mesh::Entity localElement, size_t indexConnElement) const
{
    impl::LocalId local_id = get_local_element_id(localElement);
    const GraphEdge & graphEdge = m_graph.get_edge_for_element(local_id, indexConnElement);
    const impl::parallel_info& parallelInfo = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
    return parallelInfo.m_other_proc;
}

bool ElemElemGraph::is_connected_to_other_element_via_side_ordinal(stk::mesh::Entity element, int sideOrdinal) const
{
    impl::LocalId elementLocalId = get_local_element_id(element);
    std::vector<GraphEdge> graphEdges = m_graph.get_edges_for_element_side(elementLocalId, sideOrdinal);
    return !graphEdges.empty();
}

size_t ElemElemGraph::num_edges() const
{
    return m_graph.get_num_edges();
}

impl::parallel_info& ElemElemGraph::get_parallel_edge_info(stk::mesh::Entity element, int side1, stk::mesh::EntityId remote_id, int side2)
{
    impl::LocalId this_elem_local_id = get_local_element_id(element);
    impl::LocalId elem2 = m_parallelInfoForGraphEdges.convert_remote_global_id_to_negative_local_id(remote_id);
    return m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(GraphEdge(this_elem_local_id, side1, elem2, side2));
}

const impl::parallel_info& ElemElemGraph::get_const_parallel_edge_info(stk::mesh::Entity element, int side1, stk::mesh::EntityId remote_id, int side2) const
{
    impl::LocalId this_elem_local_id = get_local_element_id(element);
    impl::LocalId elem2 = m_parallelInfoForGraphEdges.convert_remote_global_id_to_negative_local_id(remote_id);
    return m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(GraphEdge(this_elem_local_id, side1, elem2, side2));
}

impl::LocalId ElemElemGraph::get_local_element_id(stk::mesh::Entity local_element, bool require_valid_id) const
{
    ThrowRequireMsg(m_entity_to_local_id.size() > local_element.local_offset(),"Program error. Contact sierra-help@sandia.gov for support.");
    impl::LocalId local_id = m_entity_to_local_id[local_element.local_offset()];
    if (require_valid_id)
    {
        ThrowRequireMsg(local_id != impl::INVALID_LOCAL_ID, "Program error. Contact sierra-help@sandia.gov for support.");
    }
    return local_id;
}

int ElemElemGraph::size_data_members()
{
    std::vector<unsigned> counts(stk::topology::NUM_RANKS,0);
    stk::mesh::count_entities(m_bulk_data.mesh_meta_data().locally_owned_part(), m_bulk_data, counts);
    int numElems = counts[stk::topology::ELEM_RANK];

    m_graph.set_num_local_elements(numElems);
    m_local_id_to_element_entity.resize(numElems, Entity());
    m_entity_to_local_id.resize(m_bulk_data.m_entity_keys.size(), impl::INVALID_LOCAL_ID);
    m_element_topologies.resize(numElems);
    m_num_parallel_edges = 0;
    m_local_id_in_pool.resize(numElems, false);

    return numElems;
}

void ElemElemGraph::clear_data_members()
{
    m_graph.clear();
    m_local_id_to_element_entity.clear();
    m_entity_to_local_id.clear();
    m_element_topologies.clear();
    m_num_parallel_edges = 0;
    m_local_id_in_pool.clear();
}

void ElemElemGraph::resize_entity_to_local_id_if_needed(size_t maxIndexOfNewlyAddedEntities)
{
    const size_t minimumNewSize = maxIndexOfNewlyAddedEntities+1;
    const size_t oldSize = m_entity_to_local_id.size();
    const size_t newSize = std::max(minimumNewSize, oldSize);
    m_entity_to_local_id.resize(newSize, impl::INVALID_LOCAL_ID);
}

void ElemElemGraph::get_element_side_pairs(const stk::mesh::MeshIndex &meshIndex, impl::LocalId local_elem_id, std::vector<stk::mesh::GraphEdge> &graphEdges) const
{
    stk::mesh::EntityVector side_nodes;
    stk::mesh::EntityVector connected_elements;
    impl::ConnectedElementDataVector connectedElementDataVector;
    stk::mesh::Entity element = (*meshIndex.bucket)[meshIndex.bucket_ordinal];
    stk::topology topology = meshIndex.bucket->topology();
    int num_sides = topology.num_sides();
    const stk::mesh::Entity* elem_nodes = meshIndex.bucket->begin_nodes(meshIndex.bucket_ordinal);
    graphEdges.clear();
    for(int side_index=0; side_index<num_sides; ++side_index)
    {
        unsigned num_side_nodes = topology.side_topology(side_index).num_nodes();
        side_nodes.resize(num_side_nodes);
        topology.side_nodes(elem_nodes, side_index, side_nodes.begin());
        connected_elements.clear();
        connectedElementDataVector.clear();
        impl::find_locally_owned_elements_these_nodes_have_in_common(m_bulk_data, num_side_nodes, side_nodes.data(), connected_elements);
        add_local_elements_to_connected_list(connected_elements, side_nodes, connectedElementDataVector);
        impl::filter_out_invalid_solid_shell_connections(m_bulk_data, element, side_index, connectedElementDataVector);

        for (const impl::ConnectedElementData & elemData: connectedElementDataVector)
        {
            if (local_elem_id != elemData.m_elementLocalId)
            {
                graphEdges.push_back(stk::mesh::GraphEdge(local_elem_id, side_index, elemData.m_elementLocalId, elemData.m_sideIndex));
            }
        }
    }
    std::sort(graphEdges.begin(), graphEdges.end(), GraphEdgeLessByElem2());
    auto new_end = std::unique(graphEdges.begin(), graphEdges.end());
    graphEdges.resize(new_end - graphEdges.begin());
}

void ElemElemGraph::fill_graph()
{
    const stk::mesh::BucketVector& elemBuckets = m_bulk_data.get_buckets(stk::topology::ELEM_RANK, m_bulk_data.mesh_meta_data().locally_owned_part());
    for(size_t i=0; i<elemBuckets.size(); ++i)
    {
        const stk::mesh::Bucket& bucket = *elemBuckets[i];
        std::vector<stk::mesh::GraphEdge> newGraphEdges;

        for(size_t j=0; j<bucket.size(); ++j)
        {
            impl::LocalId local_elem_id = get_local_element_id(bucket[j]);
            get_element_side_pairs(stk::mesh::MeshIndex(elemBuckets[i],j), local_elem_id, newGraphEdges);
            for(const stk::mesh::GraphEdge &graphEdge : newGraphEdges)
            {
                m_graph.add_edge(graphEdge);
            }
        }
    }
}

void ElemElemGraph::fill_parallel_graph(impl::ElemSideToProcAndFaceId& elem_side_comm)
{
    stk::mesh::EntityVector elements_to_ignore;
    fill_parallel_graph(elem_side_comm, elements_to_ignore);
}

void ElemElemGraph::fill_parallel_graph(impl::ElemSideToProcAndFaceId& element_side_data_sent,
                                        const stk::mesh::EntityVector & elements_to_ignore)
{
    stk::CommSparse comm(m_bulk_data.parallel());

    size_t num_edge_ids_used = pack_shared_side_nodes_of_elements(comm, element_side_data_sent);
    m_sideIdPool.reset_suggested_side_id_iter(num_edge_ids_used);

    comm.allocate_buffers();

    pack_shared_side_nodes_of_elements(comm, element_side_data_sent);

    comm.communicate();

    stk::mesh::impl::ConnectedElementDataVector communicatedElementDataVector;
    std::map<EntityVector, stk::mesh::impl::ConnectedElementDataVector> element_side_data_received;

    for(int proc_id=0; proc_id<m_bulk_data.parallel_size(); ++proc_id)
    {
        if (proc_id != m_bulk_data.parallel_rank())
        {
            while(comm.recv_buffer(proc_id).remaining())
            {
                stk::mesh::impl::ConnectedElementData elementData;
                elementData.m_procId = proc_id;
                comm.recv_buffer(proc_id).unpack<stk::mesh::EntityId>(elementData.m_elementIdentifier);
                comm.recv_buffer(proc_id).unpack<stk::topology>(elementData.m_elementTopology);
                comm.recv_buffer(proc_id).unpack<unsigned>(elementData.m_sideIndex);
                comm.recv_buffer(proc_id).unpack<stk::mesh::EntityId>(elementData.m_suggestedFaceId);
                comm.recv_buffer(proc_id).unpack<bool>(elementData.m_isInPart);
                if(m_air_selector!=nullptr)
                {
                    comm.recv_buffer(proc_id).unpack<bool>(elementData.m_isAir);
                }

                unsigned num_side_nodes = 0;
                comm.recv_buffer(proc_id).unpack<unsigned>(num_side_nodes);
                elementData.m_sideNodes.resize(num_side_nodes);
                for(unsigned i=0; i<num_side_nodes; ++i)
                {
                    stk::mesh::EntityKey key;
                    comm.recv_buffer(proc_id).unpack<stk::mesh::EntityKey>(key);
                    elementData.m_sideNodes[i] = m_bulk_data.get_entity(key);
                }

                stk::mesh::EntityVector sortedSideNodes = elementData.m_sideNodes;
                std::sort(sortedSideNodes.begin(), sortedSideNodes.end());
                element_side_data_received[sortedSideNodes].push_back(elementData);
            }
        }
    }

    std::vector<impl::SharedEdgeInfo> newlySharedEdges;

    for (std::map<EntityVector, stk::mesh::impl::ConnectedElementDataVector>::value_type & elementData: element_side_data_received)
    {
        add_possibly_connected_elements_to_graph_using_side_nodes(element_side_data_sent, elementData.second, elements_to_ignore, newlySharedEdges);
    }

    std::vector<impl::SharedEdgeInfo> receivedSharedEdges;
    communicate_remote_edges_for_pre_existing_graph_nodes(newlySharedEdges, receivedSharedEdges);

    for (const impl::SharedEdgeInfo &receivedSharedEdge : receivedSharedEdges)
    {
        connect_remote_element_to_existing_graph( receivedSharedEdge);
    }
//    this->write_graph();
}

void ElemElemGraph::write_graph() const
{
    std::ostringstream os;
    os << "Graph for processor " << m_bulk_data.parallel_rank() << std::endl;
    for(size_t localId=0;localId<m_graph.get_num_elements_in_graph();++localId)
    {
        stk::mesh::Entity e=m_local_id_to_element_entity[localId];
        os << "Element " << m_bulk_data.identifier(e) << " has connections: ";
        for(const stk::mesh::GraphEdge &graphEdge : m_graph.get_edges_for_element(localId))
            write_graph_edge(os, graphEdge, e);
        auto iter = m_coincidentGraph.find(localId);
        if(iter != m_coincidentGraph.end())
            for(const stk::mesh::GraphEdge &graphEdge : iter->second)
                write_graph_edge(os, graphEdge, e);
        os << std::endl;
    }
    std::cerr << os.str();
}

void ElemElemGraph::write_graph_edge(std::ostringstream& os,
                                     const stk::mesh::GraphEdge& graphEdge,
                                     stk::mesh::Entity e) const
{
    stk::mesh::EntityId other;
    stk::mesh::EntityId chosenSideId = 0;
    if(graphEdge.elem2 < 0)
    {
        other = -1 * graphEdge.elem2;
        chosenSideId = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge).m_chosen_side_id;
    }
    else
    {
        other = m_bulk_data.identifier(m_local_id_to_element_entity[graphEdge.elem2]);
    }
    os << "(" << m_bulk_data.identifier(e) << ", " << graphEdge.side1 << ")<->";
    os << "(" << other << ", " << graphEdge.side2 << ")[" << chosenSideId << "] ";
}

size_t ElemElemGraph::pack_shared_side_nodes_of_elements(stk::CommSparse &comm, impl::ElemSideToProcAndFaceId& elements_to_communicate)
{
    impl::ElemSideToProcAndFaceId::iterator iter = elements_to_communicate.begin();
    impl::ElemSideToProcAndFaceId::const_iterator end = elements_to_communicate.end();
    size_t counter = 0;

    for(; iter!= end; ++iter)
    {
        stk::mesh::Entity elem = iter->first.entity;
        unsigned side_index    = iter->first.side_id;
        int sharing_proc       = iter->second.proc;
        stk::mesh::EntityId element_id     = m_bulk_data.identifier(elem);
        stk::mesh::EntityId suggested_side_id = m_sideIdPool.get_available_id();
        ++counter;
        iter->second.side_id = suggested_side_id;

        stk::topology topology = m_bulk_data.bucket(elem).topology();
        const bool isSelected = m_skinned_selector(m_bulk_data.bucket(elem));
        const stk::mesh::Entity* elem_nodes = m_bulk_data.begin_nodes(elem);

        unsigned num_nodes_this_side = topology.side_topology(side_index).num_nodes();
        stk::mesh::EntityVector side_nodes(num_nodes_this_side);
        topology.side_nodes(elem_nodes, side_index, side_nodes.begin());

        std::vector<stk::mesh::EntityKey> side_node_entity_keys(num_nodes_this_side);
        for(size_t i=0; i<num_nodes_this_side; ++i)
        {
            side_node_entity_keys[i] = m_bulk_data.entity_key(side_nodes[i]);
        }

        comm.send_buffer(sharing_proc).pack<stk::mesh::EntityId>(element_id);
        comm.send_buffer(sharing_proc).pack<stk::topology>(topology);
        comm.send_buffer(sharing_proc).pack<unsigned>(side_index);
        comm.send_buffer(sharing_proc).pack<stk::mesh::EntityId>(suggested_side_id);
        comm.send_buffer(sharing_proc).pack<bool>(isSelected);
        if(m_air_selector != nullptr)
        {
            bool is_in_air = (*m_air_selector)(m_bulk_data.bucket(elem));
            comm.send_buffer(sharing_proc).pack<bool>(is_in_air);
        }

        comm.send_buffer(sharing_proc).pack<unsigned>(num_nodes_this_side);
        for(size_t i=0; i<num_nodes_this_side; ++i)
        {
            comm.send_buffer(sharing_proc).pack<stk::mesh::EntityKey>(side_node_entity_keys[i]);
        }
    }
    return counter;
}


void ElemElemGraph::communicate_remote_edges_for_pre_existing_graph_nodes(const std::vector<impl::SharedEdgeInfo> &newlySharedEdges,
                                                                          std::vector<impl::SharedEdgeInfo> &receivedSharedEdges)
{
    stk::CommSparse comm(m_bulk_data.parallel());

    impl::pack_newly_shared_remote_edges(comm, m_bulk_data, newlySharedEdges);
    comm.allocate_buffers();

    impl::pack_newly_shared_remote_edges(comm, m_bulk_data, newlySharedEdges);
    comm.communicate();

    for(int proc_id=0; proc_id<m_bulk_data.parallel_size(); ++proc_id)
    {
        if (proc_id != m_bulk_data.parallel_rank())
        {
            while(comm.recv_buffer(proc_id).remaining())
            {
                impl::SharedEdgeInfo remoteEdgeInfo;
                remoteEdgeInfo.m_procId = proc_id;

                comm.recv_buffer(proc_id).unpack<stk::mesh::EntityId>(remoteEdgeInfo.m_remoteElementId);
                comm.recv_buffer(proc_id).unpack<stk::mesh::EntityId>(remoteEdgeInfo.m_locaElementlId);
                comm.recv_buffer(proc_id).unpack<unsigned>(remoteEdgeInfo.m_sideIndex);
                comm.recv_buffer(proc_id).unpack<stk::mesh::EntityId>(remoteEdgeInfo.m_chosenSideId);
                comm.recv_buffer(proc_id).unpack<bool>(remoteEdgeInfo.m_isInPart);
                comm.recv_buffer(proc_id).unpack<bool>(remoteEdgeInfo.m_isInAir);
                comm.recv_buffer(proc_id).unpack<stk::topology>(remoteEdgeInfo.m_remoteElementTopology);
                unsigned numNodes = 0;
                comm.recv_buffer(proc_id).unpack<unsigned>(numNodes);
                stk::mesh::EntityVector nodes;
                for(unsigned i=0; i<numNodes; ++i)
                {
                    stk::mesh::EntityKey key;
                    comm.recv_buffer(proc_id).unpack<stk::mesh::EntityKey>(key);
                    nodes.push_back(m_bulk_data.get_entity(key));
                }
                remoteEdgeInfo.m_sharedNodes = nodes;
                receivedSharedEdges.push_back(remoteEdgeInfo);
            }
        }
    }
}

void ElemElemGraph::connect_remote_element_to_existing_graph( const impl::SharedEdgeInfo &receivedSharedEdge)
{
    // If received element is a shell, it was pre-existing.  Thus, we don't have to unhook the local element graph
    // node from any remote element that has a face that shell is coincident with.

    const stk::mesh::EntityVector &sideNodes = receivedSharedEdge.m_sharedNodes;  // Pick a convenient set of side nodes (same for all elems)
    unsigned num_side_nodes = sideNodes.size();


    stk::mesh::Entity localElem = m_bulk_data.get_entity(stk::topology::ELEM_RANK, receivedSharedEdge.m_locaElementlId);
    stk::topology localElemTopology = m_bulk_data.bucket(localElem).topology();

    const stk::mesh::Entity* localElemNodes = m_bulk_data.begin_nodes(localElem);
    unsigned localElemNumSides = localElemTopology.num_sides();
    for(unsigned side_index=0; side_index<localElemNumSides; ++side_index)
    {
        unsigned num_nodes_this_side = localElemTopology.side_topology(side_index).num_nodes();
        if (num_nodes_this_side != num_side_nodes)
        {
            continue;
        }
        stk::mesh::EntityVector localElemSideNodes(num_nodes_this_side);
        localElemTopology.side_nodes(localElemNodes, side_index, localElemSideNodes.begin());

        std::pair<bool,unsigned> result = localElemTopology.side_topology(side_index).equivalent(localElemSideNodes, sideNodes);
        bool same_type_of_element = localElemTopology == receivedSharedEdge.m_remoteElementTopology;
        bool negative_permutation = result.second < localElemTopology.side_topology(side_index).num_positive_permutations();
        if ((result.first) && (same_type_of_element == negative_permutation))
        {
            impl::LocalId local_elem_id = get_local_element_id(localElem);
            impl::LocalId negSgnRemoteElemId = -1*receivedSharedEdge.m_remoteElementId;

            GraphEdge graphEdge(local_elem_id, side_index, negSgnRemoteElemId, receivedSharedEdge.m_sideIndex);
            m_graph.add_edge(graphEdge);
            stk::mesh::EntityId chosen_side_id = receivedSharedEdge.m_chosenSideId;

            impl::parallel_info parInfo(receivedSharedEdge.m_procId,
                                        result.second,
                                        chosen_side_id,
                                        receivedSharedEdge.m_remoteElementTopology,
                                        receivedSharedEdge.m_isInPart,
                                        receivedSharedEdge.m_isInAir);
            m_parallelInfoForGraphEdges.insert_parallel_info_for_graph_edge(graphEdge, parInfo);
            break;
        }
    }
}

void ElemElemGraph::filter_for_elements_in_graph(stk::mesh::EntityVector &localElements)
{
    stk::mesh::EntityVector filteredElements;
    for (stk::mesh::Entity &element : localElements)
    {
        if (is_valid_graph_element(element))
        {
            filteredElements.push_back(element);
        }
    }
    localElements.swap(filteredElements);
}


stk::mesh::EntityVector get_filtered_element_list_for_received_side_nodes(const stk::mesh::BulkData& bulk_data, const stk::mesh::impl::ConnectedElementDataVector & elemSideDataReceived, const stk::mesh::EntityVector & elements_to_ignore)
{
    const stk::mesh::EntityVector sideNodesOfReceivedElement = elemSideDataReceived[0].m_sideNodes;  // Pick a convenient set of side nodes (same for all elems)
    unsigned num_side_nodes_received_element = sideNodesOfReceivedElement.size();
    stk::mesh::EntityVector localElementsConnectedToReceivedSideNodes;
    impl::find_locally_owned_elements_these_nodes_have_in_common(bulk_data, num_side_nodes_received_element, sideNodesOfReceivedElement.data(), localElementsConnectedToReceivedSideNodes);

    // Trim down the list of local elements, ignoring elements in the provided list (since, e.g., these elements may be deleted soon)
    stk::mesh::EntityVector sortedElementsToIgnore(elements_to_ignore);
    std::sort(sortedElementsToIgnore.begin(), sortedElementsToIgnore.end());  // localElementsConnectedToReceivedSideNodes should already be sorted

    stk::mesh::EntityVector localElementsAttachedToReceivedNodes(localElementsConnectedToReceivedSideNodes.size());
    stk::mesh::EntityVector::iterator resultIter = std::set_difference(localElementsConnectedToReceivedSideNodes.begin(), localElementsConnectedToReceivedSideNodes.end(), sortedElementsToIgnore.begin(), sortedElementsToIgnore.end(), localElementsAttachedToReceivedNodes.begin());
    localElementsAttachedToReceivedNodes.resize(resultIter - localElementsAttachedToReceivedNodes.begin());

    stk::util::sort_and_unique(localElementsAttachedToReceivedNodes);
    return localElementsAttachedToReceivedNodes;
}

void add_shared_edge(const impl::ConnectedElementData& elemData, stk::mesh::EntityId elem_id, const unsigned side_index,
        const stk::mesh::EntityVector localElemSideNodes, stk::topology elem_topology, std::vector<impl::SharedEdgeInfo> &newlySharedEdges)
{
    impl::SharedEdgeInfo sharedEdgeInfo;
    sharedEdgeInfo.m_locaElementlId = elem_id;
    sharedEdgeInfo.m_remoteElementId = elemData.m_elementIdentifier;
    sharedEdgeInfo.m_procId = elemData.m_procId;
    sharedEdgeInfo.m_sideIndex = side_index;
    sharedEdgeInfo.m_sharedNodes = localElemSideNodes;
    sharedEdgeInfo.m_chosenSideId = elemData.m_suggestedFaceId;
    sharedEdgeInfo.m_isInPart = elemData.m_isInPart;
    sharedEdgeInfo.m_remoteElementTopology = elem_topology;
    sharedEdgeInfo.m_isInAir = elemData.m_isAir;
    newlySharedEdges.push_back(sharedEdgeInfo);
}

stk::mesh::EntityId find_side_id(const stk::mesh::impl::ElemSideToProcAndFaceId& elemSideDataSent,
        const stk::mesh::Entity localElem, const unsigned side_index, const int elem_proc_id)
{
    stk::mesh::EntityId chosen_side_id = stk::mesh::InvalidEntityId;
    const auto iterRange = elemSideDataSent.equal_range(impl::EntitySidePair(localElem, side_index));
    for (impl::ElemSideToProcAndFaceId::const_iterator iter = iterRange.first; iter != iterRange.second; ++iter)
    {
        bool is_received_element_data_info_from_proc_in_elem_side_comm = iter->second.proc == elem_proc_id;
        //bool is_received_element_data_info_have_same_side_index_as_in_elem_side_comm = iter->first.side_id == elemData.m_sideIndex;

        if (is_received_element_data_info_from_proc_in_elem_side_comm)
        {
            chosen_side_id = iter->second.side_id;
            break;
        }
    }
    ThrowRequireMsg(chosen_side_id!=stk::mesh::InvalidEntityId, "Program error. Please contact sierra-help@sandia.gov for support");
    return chosen_side_id;
}

stk::topology ElemElemGraph::get_topology_of_remote_element(const GraphEdge &graphEdge)
{
    impl::parallel_info &parallel_edge_info = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
    return parallel_edge_info.m_remote_element_toplogy;
}

stk::topology ElemElemGraph::get_topology_of_connected_element(const GraphEdge &graphEdge)
{
    if(graphEdge.elem2<0)
    {
        return this->get_topology_of_remote_element(graphEdge);
    }
    return m_element_topologies[graphEdge.elem2];
}

void ElemElemGraph::add_possibly_connected_elements_to_graph_using_side_nodes( const stk::mesh::impl::ElemSideToProcAndFaceId& elemSideDataSent,
                                                                               stk::mesh::impl::ConnectedElementDataVector & elemSideDataReceived,
                                                                               const stk::mesh::EntityVector & elements_to_ignore,
                                                                               std::vector<impl::SharedEdgeInfo> &newlySharedEdges)
{
    if (elemSideDataReceived.empty()) {
        return;  // Nothing to connect to
    }

    // Could be the localElements are discoverable by using elemSideDataSent?

    stk::mesh::EntityVector localElementsAttachedToReceivedNodes =
            get_filtered_element_list_for_received_side_nodes(m_bulk_data, elemSideDataReceived, elements_to_ignore);

    stk::mesh::EntityVector sideNodesOfReceivedElement = elemSideDataReceived[0].m_sideNodes;  // Pick a convenient set of side nodes (same for all elems)
    unsigned num_side_nodes_received_element = sideNodesOfReceivedElement.size();

    add_local_elements_to_connected_list(localElementsAttachedToReceivedNodes, sideNodesOfReceivedElement, elemSideDataReceived);

    std::set<EntityId> localElementsConnectedToRemoteShell;
    std::vector<std::pair<stk::mesh::Entity, stk::mesh::EntityId>> localAndRemoteElementsConnectedToShell;
    for (stk::mesh::Entity localElem: localElementsAttachedToReceivedNodes)
    {
        stk::topology localElemTopology = m_bulk_data.bucket(localElem).topology();

        const stk::mesh::Entity* localElemNodes = m_bulk_data.begin_nodes(localElem);
        unsigned localElemNumSides = localElemTopology.num_sides();
        for(unsigned side_index=0; side_index<localElemNumSides; ++side_index)
        {
            unsigned num_nodes_this_side = localElemTopology.side_topology(side_index).num_nodes();
            if (num_nodes_this_side == num_side_nodes_received_element)
            {
                stk::mesh::EntityVector localElemSideNodes(num_nodes_this_side);
                localElemTopology.side_nodes(localElemNodes, side_index, localElemSideNodes.begin());

                impl::ConnectedElementDataVector filteredCommunicatedElementData = elemSideDataReceived;  // Modify copy for this local element
                impl::filter_out_invalid_solid_shell_connections(m_bulk_data, localElem, side_index, filteredCommunicatedElementData);

                // Loop thru all elements left over to see if they are connected to local elem
                for (const impl::ConnectedElementData & elemDataFromOtherProc : filteredCommunicatedElementData)
                {
                    const bool isRemoteElement = (elemDataFromOtherProc.m_procId != m_bulk_data.parallel_rank());
                    std::pair<bool,unsigned> result = localElemTopology.side_topology(side_index).equivalent(localElemSideNodes, elemDataFromOtherProc.m_sideNodes);
                    bool are_local_element_and_remote_element_connected_via_side_nodes = result.first;
                    if (isRemoteElement && are_local_element_and_remote_element_connected_via_side_nodes)
                    {
                        impl::LocalId local_elem_id = get_local_element_id(localElem);
                        if (elemDataFromOtherProc.m_elementTopology.is_shell())
                        {
                            localElementsConnectedToRemoteShell.insert(local_elem_id);  // Store connection to error-check later
                        }

                        stk::mesh::EntityId chosen_side_id = 0;
                        impl::LocalId negSgnRemoteElemId = -1*elemDataFromOtherProc.m_elementIdentifier;
                        stk::mesh::GraphEdge graphEdge(local_elem_id, side_index, negSgnRemoteElemId, elemDataFromOtherProc.m_sideIndex);
                        m_graph.add_edge(graphEdge);

                        stk::mesh::impl::ElemSideToProcAndFaceId::const_iterator iter_elem_side_comm = elemSideDataSent.find(impl::EntitySidePair(localElem, side_index));
                        if(iter_elem_side_comm != elemSideDataSent.end())
                        {
                            if(m_bulk_data.identifier(localElem) < elemDataFromOtherProc.m_elementIdentifier)
                            {
                                chosen_side_id = find_side_id(elemSideDataSent, localElem, side_index, elemDataFromOtherProc.m_procId);
                            }
                            else
                            {
                                chosen_side_id = elemDataFromOtherProc.m_suggestedFaceId;
                            }
                        }
                        else
                        {
                            add_shared_edge(elemDataFromOtherProc, m_bulk_data.identifier(localElem), side_index, localElemSideNodes,
                                    m_bulk_data.bucket(localElem).topology(), newlySharedEdges);
                        }

                        int thisElemSidePermutation = result.second;
                        impl::parallel_info parInfo(elemDataFromOtherProc.m_procId,
                                                    thisElemSidePermutation,
                                                    chosen_side_id,
                                                    elemDataFromOtherProc.m_elementTopology,
                                                    elemDataFromOtherProc.m_isInPart,
                                                    elemDataFromOtherProc.m_isAir);

                        m_parallelInfoForGraphEdges.insert_parallel_info_for_graph_edge(graphEdge, parInfo);
                    }
                }
            }
        }
    }
}

void report_error_with_invalid_ordinal(std::pair<stk::mesh::ConnectivityOrdinal, stk::mesh::Permutation> ord_and_perm, const stk::mesh::BulkData& bulkData, const stk::mesh::EntityVector& side_nodes_vec,
        stk::mesh::Entity element_with_perm_0, stk::mesh::Entity element_with_perm_4)
{
    if(ord_and_perm.first == stk::mesh::INVALID_CONNECTIVITY_ORDINAL)
    {
        std::ostringstream os;
        os << "Proc: " << bulkData.parallel_rank() << std::endl;
        os << "this element: " << bulkData.identifier(element_with_perm_0) << std::endl;
        os << "other element: " << bulkData.identifier(element_with_perm_4) << std::endl;
        os << "Nodes: ";

        for(stk::mesh::Entity side_node : side_nodes_vec)
        {
            os << bulkData.identifier(side_node) << " ";
        }

        os << std::endl;
        std::cerr << os.str();
    }

    ThrowRequireMsg(ord_and_perm.first != stk::mesh::INVALID_CONNECTIVITY_ORDINAL, "yikes!");
    ThrowRequireMsg(ord_and_perm.second != stk::mesh::INVALID_PERMUTATION, "yikes!");
}

void ensure_fresh_modifiable_state(stk::mesh::BulkData& bulkData)
{
    if(bulkData.in_modifiable_state())
    {
        bulkData.modification_end();
    }
    bulkData.modification_begin();
}

class RemoteDeathBoundary
{
public:
    RemoteDeathBoundary(stk::mesh::BulkData& bulkData, ElemElemGraph& elementGraph,
        const stk::mesh::EntityVector& killedElements, const stk::mesh::PartVector& parts_for_creating_side, stk::mesh::Part& active, const stk::mesh::PartVector* boundary_mesh_parts) :
            m_bulkData(bulkData), m_elementGraph(elementGraph), m_killedElements(killedElements), m_parts_for_creating_side(parts_for_creating_side), m_active(active),
            m_boundary_mesh_parts(boundary_mesh_parts), m_topology_modified(false)
    {}
    ~RemoteDeathBoundary(){}

    void update_death_boundary_for_remotely_killed_elements(std::vector<stk::mesh::sharing_info> &shared_modified, stk::mesh::EntityVector& deletedEntities)
    {
        std::vector<impl::graphEdgeProc> remote_edges = get_remote_edges();

        for(impl::graphEdgeProc& re : remote_edges)
        {
            stk::mesh::EntityId local_id = re.m_localElementId;
            int local_side = re.m_localSide;
            stk::mesh::EntityId remote_id = re.m_remoteElementId;
            int remote_side = re.m_remoteSide;

            stk::mesh::Entity element = m_bulkData.get_entity(stk::topology::ELEM_RANK, local_id);

            impl::parallel_info &parallel_edge_info = m_elementGraph.get_parallel_edge_info(element, local_side, remote_id, remote_side);
            parallel_edge_info.m_in_body_to_be_skinned = false;

            m_topology_modified = true;

            bool create_side = m_bulkData.bucket(element).member(m_active);
            if(create_side==true)
            {
                impl::add_side_into_exposed_boundary(m_bulkData, parallel_edge_info, element, local_side, remote_id, m_parts_for_creating_side,
                        shared_modified, m_boundary_mesh_parts);
            }
            else
            {
                impl::remove_side_from_death_boundary(m_bulkData, element, m_active, deletedEntities, local_side);
            }
        }
    }

    void set_topology_is_modified()
    {
        m_topology_modified = true;
    }

    bool get_topology_modification_status() const
    {
        return m_topology_modified;
    }

private:

    std::vector<impl::graphEdgeProc> get_remote_edges() const
    {
        std::vector<impl::graphEdgeProc> elements_to_comm = get_elements_to_communicate();
        return impl::communicate_killed_entities(m_bulkData.parallel(), elements_to_comm);
    }

    std::vector<impl::graphEdgeProc> get_elements_to_communicate() const
    {
        std::vector<impl::graphEdgeProc> elements_to_comm;

        for(stk::mesh::Entity this_element :m_killedElements)
        {
            for(size_t j=0;j<m_elementGraph.get_num_connected_elems(this_element);++j)
            {
                if(impl::does_element_have_side(m_bulkData, this_element) && !m_elementGraph.is_connected_elem_locally_owned(this_element, j))
                {
                    impl::IdViaSidePair idViaSidePair = m_elementGraph.get_connected_remote_id_and_via_side(this_element,j);
                    stk::mesh::EntityId other_element_id = idViaSidePair.id;
                    int side1 = idViaSidePair.side;
                    int side2 = m_elementGraph.get_connected_elements_side(this_element,j);
                    int other_proc = m_elementGraph.get_owning_proc_id_of_remote_element(this_element, j);
                    elements_to_comm.push_back(impl::graphEdgeProc(m_bulkData.identifier(this_element), side1, other_element_id, side2, other_proc));
                }
            }
        }

        return elements_to_comm;
    }

    stk::mesh::BulkData& m_bulkData;
    ElemElemGraph& m_elementGraph;
    const stk::mesh::EntityVector& m_killedElements;
    const stk::mesh::PartVector& m_parts_for_creating_side;
    stk::mesh::Part& m_active;
    const stk::mesh::PartVector* m_boundary_mesh_parts;
    bool m_topology_modified;
};


bool process_killed_elements(stk::mesh::BulkData& bulkData,
                             ElemElemGraph& elementGraph,
                             const stk::mesh::EntityVector& killedElements,
                             stk::mesh::Part& active,
                             const stk::mesh::PartVector& parts_for_creating_side,
                             const stk::mesh::PartVector* boundary_mesh_parts)
{
    ensure_fresh_modifiable_state(bulkData);
    impl::create_sides_created_during_death_part(bulkData.mesh_meta_data());

    std::vector<stk::mesh::sharing_info> shared_modified;
    stk::mesh::EntityVector deletedEntities;

    RemoteDeathBoundary remote_death_boundary(bulkData, elementGraph, killedElements, parts_for_creating_side, active, boundary_mesh_parts);
    remote_death_boundary.update_death_boundary_for_remotely_killed_elements(shared_modified, deletedEntities);

    std::vector<impl::ElementSidePair> element_side_pairs;
    element_side_pairs.reserve(impl::get_element_side_multiplier() * killedElements.size());

    for(size_t k = 0; k < killedElements.size(); ++k)
    {
        stk::mesh::Entity this_element = killedElements[k];

        for(size_t j = 0; j < elementGraph.get_num_connected_elems(this_element); ++j)
        {
            if(impl::does_element_have_side(bulkData, this_element))
            {
                remote_death_boundary.set_topology_is_modified();
                if(elementGraph.is_connected_elem_locally_owned(this_element, j))
                {
                    impl::ElementViaSidePair other_element_via_side = elementGraph.get_connected_element_and_via_side(this_element, j);
                    stk::mesh::Entity other_element = other_element_via_side.element;
                    if(impl::does_element_have_side(bulkData, other_element_via_side.element))
                    {
                        int side_id = other_element_via_side.side;
                        ThrowRequireMsg(side_id != -1, "Program error. Please contact sierra-help@sandia.gov for support.");

                        bool is_other_element_alive = bulkData.bucket(other_element).member(active);
                        if(is_other_element_alive)
                        {
                            std::string msg = "Program error. Please contact sierra-help@sandia.gov for support.";

                            stk::mesh::Entity side = stk::mesh::impl::get_side_for_element(bulkData, this_element, side_id);

                            if(bulkData.is_valid(side))
                            {
                                if(bulkData.bucket(side).owned())
                                {
                                    stk::mesh::PartVector parts = impl::get_stk_parts_for_moving_parts_into_death_boundary(boundary_mesh_parts);
                                    bulkData.change_entity_parts(side, parts, stk::mesh::PartVector());
                                }
                            }
                            else
                            {
                                stk::mesh::PartVector parts = impl::get_parts_for_creating_side(bulkData, parts_for_creating_side, other_element, side_id);

                                stk::mesh::EntityId side_global_id = elementGraph.get_available_side_id();

                                stk::mesh::EntityRank side_rank = bulkData.mesh_meta_data().side_rank();
                                ThrowRequireMsg(!impl::is_id_already_in_use_locally(bulkData, side_rank, side_global_id), msg);

                                // switch elements
                                stk::mesh::Entity element_with_perm_0 = other_element;
                                stk::mesh::Entity element_with_perm_4 = this_element;

                                int side_id_needed = elementGraph.get_connected_elements_side(this_element, j);

                                ThrowRequireMsg(side_id_needed >= 0, "ERROR: proc " << bulkData.parallel_rank() << " found side_id_needed=" << side_id_needed
                                                << " between elem " << bulkData.identifier(element_with_perm_0)<< " and " << bulkData.identifier(element_with_perm_4)
                                                << " in elem-elem-graph");

                                side = stk::mesh::declare_element_side(bulkData, side_global_id, element_with_perm_0, side_id_needed, parts);

                                const stk::mesh::Entity* side_nodes = bulkData.begin_nodes(side);
                                unsigned num_side_nodes = bulkData.num_nodes(side);
                                stk::mesh::EntityVector side_nodes_vec(side_nodes, side_nodes + num_side_nodes);

                                std::pair<stk::mesh::ConnectivityOrdinal, stk::mesh::Permutation> ord_and_perm =
                                        stk::mesh::get_ordinal_and_permutation(bulkData, element_with_perm_4, side_rank, side_nodes_vec);

                                report_error_with_invalid_ordinal(ord_and_perm, bulkData, side_nodes_vec, element_with_perm_0, element_with_perm_4);

                                bulkData.declare_relation(element_with_perm_4, side, ord_and_perm.first, ord_and_perm.second);
                            }
                        }
                        else
                        {
                            impl::remove_side_from_death_boundary(bulkData, this_element, active, deletedEntities, side_id);
                        }
                    }
                }
                else
                {
                    impl::IdViaSidePair remote_id_side_pair = elementGraph.get_connected_remote_id_and_via_side(this_element, j);
                    stk::mesh::EntityId remote_id = remote_id_side_pair.id;
                    int remote_side = elementGraph.get_connected_elements_side(this_element, j);
                    impl::parallel_info &parallel_edge_info = elementGraph.get_parallel_edge_info(this_element, remote_id_side_pair.side, remote_id, remote_side);
                    bool other_element_active = parallel_edge_info.m_in_body_to_be_skinned;
                    bool create_side = other_element_active;

                    if(create_side)
                    {
                        impl::add_side_into_exposed_boundary(bulkData, parallel_edge_info, this_element, remote_id_side_pair.side, remote_id, parts_for_creating_side,
                                shared_modified, boundary_mesh_parts);
                    }
                    else
                    {
                        int side_id = remote_id_side_pair.side;
                        ThrowRequireMsg(side_id != -1, "Program error. Please contact sierra-help@sandia.gov for support.");
                        impl::remove_side_from_death_boundary(bulkData, this_element, active, deletedEntities, side_id);
                    }
                }
            }
        }
    }
    stk::mesh::impl::delete_entities_and_upward_relations(bulkData, deletedEntities);
    bulkData.make_mesh_parallel_consistent_after_element_death(shared_modified, deletedEntities, elementGraph, killedElements, active);
    return remote_death_boundary.get_topology_modification_status();
}

stk::mesh::SideConnector ElemElemGraph::get_side_connector()
{
    return stk::mesh::SideConnector(m_bulk_data, m_graph, m_coincidentGraph, m_local_id_to_element_entity, m_entity_to_local_id);
}

void ElemElemGraph::add_local_elements_to_connected_list(const stk::mesh::EntityVector & local_elements_attached_to_side_nodes,
                                                         const stk::mesh::EntityVector & side_nodes_received,
                                                         impl::ConnectedElementDataVector & element_data_received) const
{
    impl::ConnectedElementData elemData;
    stk::mesh::EntityVector connectedSideNodes;
    for (const stk::mesh::Entity & local_element: local_elements_attached_to_side_nodes)
    {
        elemData.m_sideNodes.clear();
        connectedSideNodes.clear();
        stk::mesh::OrdinalAndPermutation connectedOrdAndPerm = stk::mesh::get_ordinal_and_permutation(m_bulk_data, local_element, m_bulk_data.mesh_meta_data().side_rank(), side_nodes_received);
        if (INVALID_CONNECTIVITY_ORDINAL == connectedOrdAndPerm.first)
        {
            continue;
        }
        const stk::mesh::Bucket & connectedBucket = m_bulk_data.bucket(local_element);
        const stk::mesh::Entity* connectedElemNodes = m_bulk_data.begin_nodes(local_element);

        elemData.m_procId = m_bulk_data.parallel_rank();
        elemData.m_elementLocalId = get_local_element_id(local_element, false);
        if (elemData.m_elementLocalId == impl::INVALID_LOCAL_ID)
        {
            continue;
        }
        elemData.m_elementIdentifier = m_bulk_data.identifier(local_element);
        elemData.m_elementTopology = connectedBucket.topology();
        elemData.m_sideIndex = connectedOrdAndPerm.first;
        elemData.m_suggestedFaceId = 0;
        elemData.m_sideNodes.resize(side_nodes_received.size());
        ThrowAssertMsg(elemData.m_elementTopology.side_topology(elemData.m_sideIndex).num_nodes() == side_nodes_received.size(),
                       "Error, number of nodes on sides of adjacent elements do not agree:  " << side_nodes_received.size() << " != " << elemData.m_elementTopology.side_topology(elemData.m_sideIndex).num_nodes());

        elemData.m_elementTopology.side_nodes(connectedElemNodes, elemData.m_sideIndex, elemData.m_sideNodes.begin());
        element_data_received.push_back(elemData);

        if (elemData.m_elementTopology.is_shell())
        {
            // Also add the back-side face if this is a shell
            elemData.m_sideIndex = (elemData.m_sideIndex == 0) ? 1 : 0;
            elemData.m_elementTopology.side_nodes(connectedElemNodes, elemData.m_sideIndex, elemData.m_sideNodes.begin());
            element_data_received.push_back(elemData);
        }
    }
}

impl::LocalId ElemElemGraph::create_new_local_id(stk::mesh::Entity new_elem)
{
    if (m_entity_to_local_id.size() > new_elem.local_offset() && m_entity_to_local_id[new_elem.local_offset()] != impl::INVALID_LOCAL_ID)
    {
        return m_entity_to_local_id[new_elem.local_offset()];
    }

    impl::LocalId new_local_id = m_graph.get_num_elements_in_graph();
    if (m_deleted_element_local_id_pool.size() > 0)
    {
        new_local_id = m_deleted_element_local_id_pool.back();
        m_local_id_in_pool[new_local_id] = false;
        m_deleted_element_local_id_pool.pop_back();
    }
    else
    {
        if (m_local_id_to_element_entity.size() <= static_cast<size_t> (new_local_id))
        {
            m_local_id_to_element_entity.resize(new_local_id+1);
        }
        if (m_entity_to_local_id.size() <= new_elem.local_offset())
        {
            m_entity_to_local_id.resize(new_elem.local_offset()+1);
        }
        m_graph.add_new_element();
    }
    m_local_id_to_element_entity[new_local_id] = new_elem;
    m_entity_to_local_id[new_elem.local_offset()] = new_local_id;

    return new_local_id;
}

impl::LocalId ElemElemGraph::get_new_local_element_id_from_pool()
{
    impl::LocalId new_local_id;
    if (!m_deleted_element_local_id_pool.empty())
    {
        new_local_id = m_deleted_element_local_id_pool.back();
        m_deleted_element_local_id_pool.pop_back();
        m_local_id_in_pool[new_local_id] = false;
    }
    else
    {
        new_local_id = m_graph.get_num_elements_in_graph();
        m_graph.add_new_element();
        m_local_id_in_pool.push_back(false);
        m_local_id_to_element_entity.push_back(Entity());
        m_element_topologies.push_back(stk::topology::INVALID_TOPOLOGY);
    }
    return new_local_id;
}

bool ElemElemGraph::is_valid_graph_element(stk::mesh::Entity local_element) const
{
    bool value = false;
//    if (m_bulk_data.is_valid(local_element))
    {
        impl::LocalId max_elem_id = static_cast<impl::LocalId>(m_graph.get_num_elements_in_graph());
        if (local_element.local_offset() < m_entity_to_local_id.size())
        {
            impl::LocalId elem_id = get_local_element_id(local_element, false);
            value = elem_id >= 0 && elem_id < max_elem_id && !m_local_id_in_pool[elem_id];
        }
    }
    return value;
}

void ElemElemGraph::pack_deleted_element_comm(stk::CommSparse &comm,
                                              const std::vector<impl::DeletedElementData> &local_elem_and_remote_connected_elem)
{
    for(size_t i=0;i<local_elem_and_remote_connected_elem.size();++i)
    {
        int remote_proc = local_elem_and_remote_connected_elem[i].m_remoteProc;

        impl::LocalId elem_to_delete_local_id = local_elem_and_remote_connected_elem[i].m_deletedElement;
        stk::mesh::Entity elem_to_delete = m_local_id_to_element_entity[elem_to_delete_local_id];
        stk::mesh::EntityId elem_to_delete_global_id = m_bulk_data.identifier(elem_to_delete);
        stk::mesh::EntityId connected_elem_global_id = local_elem_and_remote_connected_elem[i].m_remoteElement;

        comm.send_buffer(remote_proc).pack<stk::mesh::EntityId>(elem_to_delete_global_id);
        comm.send_buffer(remote_proc).pack<stk::mesh::EntityId>(connected_elem_global_id);
    }
}

void ElemElemGraph::pack_shell_connectivity(stk::CommSparse & comm,
                                            const std::vector<impl::ShellConnectivityData> & shellConnectivityList,
                                            const std::vector<impl::ElementSidePair> &deletedShells)
{
    for(size_t i=0; i<shellConnectivityList.size(); i++)
    {
        const impl::ShellConnectivityData & shellConnectivityData = shellConnectivityList[i];
        if (shellConnectivityData.m_farElementIsRemote) {
            impl::LocalId deletedElementId = deletedShells[i].first;
            int deletedElementSide = deletedShells[i].second;
            GraphEdge graphEdge(deletedElementId, deletedElementSide, m_parallelInfoForGraphEdges.convert_remote_global_id_to_negative_local_id(shellConnectivityData.m_farElementId), shellConnectivityData.m_farElementSide);
            const impl::parallel_info& parallelInfo = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
            int remote_proc = parallelInfo.m_other_proc;

            comm.send_buffer(remote_proc).pack<stk::mesh::EntityId>(shellConnectivityData.m_nearElementId);
            comm.send_buffer(remote_proc).pack<int>(shellConnectivityData.m_nearElementSide);
            comm.send_buffer(remote_proc).pack<int>(shellConnectivityData.m_nearElementProc);
            comm.send_buffer(remote_proc).pack<stk::mesh::EntityId>(shellConnectivityData.m_shellElementId);
            comm.send_buffer(remote_proc).pack<stk::mesh::EntityId>(shellConnectivityData.m_farElementId);
            comm.send_buffer(remote_proc).pack<int>(shellConnectivityData.m_farElementSide);
        }
    }
}

void ElemElemGraph::collect_local_shell_connectivity_data(const stk::mesh::impl::DeletedElementInfoVector &elements_to_delete,
                                                          std::vector<impl::ShellConnectivityData>& shellConnectivityList,
                                                          std::vector<impl::ElementSidePair> &deletedShells)
{
    for (const stk::mesh::impl::DeletedElementInfo &deletedElementInfo : elements_to_delete) {
//        ThrowRequireMsg(m_bulk_data.is_valid(elem_to_delete), "Do not delete entities before removing from ElemElemGraph. Contact sierra-help@sandia.gov for support.");
        stk::mesh::Entity elem_to_delete = deletedElementInfo.entity;
        ThrowRequireMsg(is_valid_graph_element(elem_to_delete), "Program error. Not valid graph element. Contact sierra-help@sandia.gov for support.");

        if (deletedElementInfo.isShell) {
            impl::LocalId shell_to_delete_id = get_local_element_id( elem_to_delete);
            size_t num_connected_elems = m_graph.get_num_edges_for_element(shell_to_delete_id);
            ThrowAssertMsg(num_connected_elems <= 2, "Coincident shells detected.  Please call (505)844-3041 for support.");
            for (int near_elem_index = num_connected_elems - 1; near_elem_index >= 0; --near_elem_index) {
                impl::ShellConnectivityData shellConnectivityData;

                const GraphEdge & graphEdge = m_graph.get_edge_for_element(shell_to_delete_id, near_elem_index);
                impl::LocalId near_elem_id = graphEdge.elem2;

                if (near_elem_id >= 0) {
                    stk::mesh::Entity nearElem = m_local_id_to_element_entity[near_elem_id];
                    shellConnectivityData.m_nearElementSide = graphEdge.side2;
                    shellConnectivityData.m_nearElementId = m_bulk_data.identifier(nearElem);
                    shellConnectivityData.m_nearElementProc = m_bulk_data.parallel_rank();
                }
                else {
                    shellConnectivityData.m_nearElementId = -near_elem_id;

                    impl::parallel_info& parallelInfo = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
                    shellConnectivityData.m_nearElementSide = graphEdge.side2;
                    shellConnectivityData.m_nearElementProc = parallelInfo.m_other_proc;
                }

                shellConnectivityData.m_shellElementId = deletedElementInfo.identifier;

                for(size_t i=0; i<num_connected_elems; ++i) {
                    const GraphEdge & nestedGraphEdge = m_graph.get_edge_for_element(shell_to_delete_id, i);
                    impl::LocalId graphElemId = nestedGraphEdge.elem2;
                    if (graphElemId != near_elem_id) {
                        if (graphElemId < 0) {
                            // Remote Element
                            shellConnectivityData.m_farElementId = -graphElemId;
                            shellConnectivityData.m_farElementIsRemote = true;

                            impl::parallel_info& parallelInfo = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(nestedGraphEdge);
                            shellConnectivityData.m_farElementSide = nestedGraphEdge.side2;
                            shellConnectivityData.m_farElementProc = parallelInfo.m_other_proc;
                        }
                        else {
                            // Local Element
                            stk::mesh::Entity remoteElement = m_local_id_to_element_entity[graphElemId];
                            shellConnectivityData.m_farElementId = m_bulk_data.identifier(remoteElement);
                            shellConnectivityData.m_farElementSide = nestedGraphEdge.side2;
                            shellConnectivityData.m_farElementIsRemote = false;
                            shellConnectivityData.m_farElementProc = m_bulk_data.parallel_rank();
                        }
                        // Only add to list if there *is* something connected on the back-side of the shell
                        deletedShells.push_back(std::make_pair(nestedGraphEdge.elem1,nestedGraphEdge.side1));
                        shellConnectivityList.push_back(shellConnectivityData);
                        break;
                    }
                }
            }
        }
    }
}

void ElemElemGraph::communicate_shell_connectivity(std::vector<impl::ShellConnectivityData>& shellConnectivityList,
                                                   const std::vector<impl::ElementSidePair> &deletedShells) {

    stk::CommSparse shellComm(m_bulk_data.parallel());
    pack_shell_connectivity(shellComm, shellConnectivityList, deletedShells);
    shellComm.allocate_buffers();
    pack_shell_connectivity(shellComm, shellConnectivityList, deletedShells);
    shellComm.communicate();

    for (int proc = 0; proc < m_bulk_data.parallel_size(); ++proc) {
        while (shellComm.recv_buffer(proc).remaining()) {
            impl::ShellConnectivityData shellConnectivityData;
            shellComm.recv_buffer(proc).unpack<stk::mesh::EntityId>(shellConnectivityData.m_farElementId); // Flip remote and local
            shellComm.recv_buffer(proc).unpack<int>(shellConnectivityData.m_farElementSide);
            shellComm.recv_buffer(proc).unpack<int>(shellConnectivityData.m_farElementProc);
            shellComm.recv_buffer(proc).unpack<stk::mesh::EntityId>(shellConnectivityData.m_shellElementId);
            shellComm.recv_buffer(proc).unpack<stk::mesh::EntityId>(shellConnectivityData.m_nearElementId); // Flip remote and local
            shellComm.recv_buffer(proc).unpack<int>(shellConnectivityData.m_nearElementSide);
            shellConnectivityData.m_nearElementProc = m_bulk_data.parallel_rank();

            shellConnectivityList.push_back(shellConnectivityData);
        }
    }
}

void ElemElemGraph::delete_local_connections_and_collect_remote(const stk::mesh::impl::DeletedElementInfoVector &elements_to_delete,
                                                                std::vector<impl::DeletedElementData>& local_elem_and_remote_connected_elem)
{
    for (const stk::mesh::impl::DeletedElementInfo &deletedElementInfo : elements_to_delete) {
        stk::mesh::Entity elem_to_delete = deletedElementInfo.entity;
        impl::LocalId elem_to_delete_id = get_local_element_id(elem_to_delete);

        size_t num_connected_elems = m_graph.get_num_edges_for_element(elem_to_delete_id);
        for (int conn_elem_index = num_connected_elems - 1; conn_elem_index >= 0; --conn_elem_index) {
            const GraphEdge & graphEdge = m_graph.get_edge_for_element(elem_to_delete_id, conn_elem_index);
            impl::LocalId connected_elem_id = graphEdge.elem2;

            bool local_connection = connected_elem_id >= 0;
            if (local_connection) {
                m_graph.delete_edge(create_symmetric_edge(graphEdge));
            } else {
                impl::parallel_info& parallelInfo = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
                int remote_proc = parallelInfo.m_other_proc;

                stk::mesh::EntityId connected_global_id = -connected_elem_id;
                impl::DeletedElementData deletedElementData;
                deletedElementData.m_deletedElement = elem_to_delete_id;
                deletedElementData.m_remoteElement = connected_global_id;
                deletedElementData.m_remoteProc = remote_proc;
                local_elem_and_remote_connected_elem.push_back(deletedElementData);
                m_parallelInfoForGraphEdges.erase_parallel_info_for_graph_edge(graphEdge);
            }
        }

        m_graph.delete_all_connections(elem_to_delete_id);
    }
}

void ElemElemGraph::communicate_remote_connections_to_delete(const std::vector<impl::DeletedElementData>& local_elem_and_remote_connected_elem,
                                                             std::vector<std::pair<stk::mesh::EntityId, stk::mesh::EntityId> >& remote_edges)
{
    stk::CommSparse comm(m_bulk_data.parallel());
    pack_deleted_element_comm(comm, local_elem_and_remote_connected_elem);
    comm.allocate_buffers();
    pack_deleted_element_comm(comm, local_elem_and_remote_connected_elem);
    comm.communicate();

    for (int proc = 0; proc < m_bulk_data.parallel_size(); ++proc) {
        while (comm.recv_buffer(proc).remaining()) {
            stk::mesh::EntityId deleted_elem_global_id;
            stk::mesh::EntityId connected_elem_global_id;
            comm.recv_buffer(proc).unpack<stk::mesh::EntityId>(deleted_elem_global_id);
            comm.recv_buffer(proc).unpack<stk::mesh::EntityId>(connected_elem_global_id);
            remote_edges.push_back(std::make_pair(deleted_elem_global_id, connected_elem_global_id));
        }
    }
}

void ElemElemGraph::clear_deleted_element_connections(const stk::mesh::impl::DeletedElementInfoVector &elements_to_delete)
{
    for (const stk::mesh::impl::DeletedElementInfo &deletedElementInfo : elements_to_delete)
    {
        stk::mesh::Entity elem = deletedElementInfo.entity;
        impl::LocalId elem_to_delete_id = m_entity_to_local_id[elem.local_offset()];
        m_deleted_element_local_id_pool.push_back(elem_to_delete_id);
        m_local_id_in_pool[elem_to_delete_id] = true;
        m_element_topologies[elem_to_delete_id] = stk::topology::INVALID_TOPOLOGY;
        m_entity_to_local_id[elem.local_offset()] = impl::INVALID_LOCAL_ID;
        m_local_id_to_element_entity[elem_to_delete_id] = stk::mesh::Entity::InvalidEntity;
    }
}

void ElemElemGraph::delete_remote_connections(const std::vector<std::pair<stk::mesh::EntityId, stk::mesh::EntityId> >& remote_edges)
{
    for (const std::pair<stk::mesh::EntityId, stk::mesh::EntityId> & edge : remote_edges) {
        stk::mesh::EntityId deleted_elem_global_id = edge.first;
        stk::mesh::EntityId connected_elem_global_id = edge.second;
        stk::mesh::Entity connected_elem = m_bulk_data.get_entity(stk::topology::ELEM_RANK, connected_elem_global_id);
        impl::LocalId connected_elem_id = m_entity_to_local_id[connected_elem.local_offset()];
        if (!is_valid_graph_element(connected_elem)) {
            continue;
        }

        size_t num_conn_elem = m_graph.get_num_edges_for_element(connected_elem_id);
        bool found_deleted_elem = false;
        for (size_t conn_elem_index = 0; conn_elem_index < num_conn_elem; ++conn_elem_index) {
            const GraphEdge & graphEdge = m_graph.get_edge_for_element(connected_elem_id, conn_elem_index);
            if (graphEdge.elem2 == static_cast<int64_t>(-deleted_elem_global_id)) {
                m_parallelInfoForGraphEdges.erase_parallel_info_for_graph_edge(graphEdge);
                m_graph.delete_edge_from_graph(connected_elem_id, conn_elem_index);
                found_deleted_elem = true;
                break;
            }
        }
        ThrowRequireMsg(found_deleted_elem, "Error. Contact sierra-help@sandia.gov for support.");
    }
}

void ElemElemGraph::reconnect_volume_elements_across_deleted_shells(std::vector<impl::ShellConnectivityData> & shellConnectivityList)
{
    // Prune the shell connectivity list for entries that we do not need to act on
    std::vector<impl::ShellConnectivityData>::iterator it = shellConnectivityList.begin();
    while (it != shellConnectivityList.end()) {
        const impl::ShellConnectivityData& shellConnectivityData = *it;
        if (shellConnectivityData.m_nearElementProc != m_bulk_data.parallel_rank()) {
            it = shellConnectivityList.erase(it);
        } else {
            ++it;
        }
    }

    impl::ElemSideToProcAndFaceId shellNeighborsToReconnect;
    for (const impl::ShellConnectivityData& data : shellConnectivityList) {
        stk::mesh::Entity localElement = m_bulk_data.get_entity(stk::topology::ELEM_RANK, data.m_nearElementId);
        if (data.m_nearElementProc == m_bulk_data.parallel_rank() && data.m_farElementProc == m_bulk_data.parallel_rank()) {
            impl::LocalId localElementId = get_local_element_id(localElement);
            stk::mesh::Entity remoteElement = m_bulk_data.get_entity(stk::topology::ELEM_RANK, data.m_farElementId);
            impl::LocalId remoteElementId;
            remoteElementId = get_local_element_id(remoteElement);
            m_graph.add_edge(stk::mesh::GraphEdge(localElementId, data.m_nearElementSide, remoteElementId, data.m_farElementSide));
        }
        else {
            shellNeighborsToReconnect.insert(
                    std::pair<impl::EntitySidePair, impl::ProcFaceIdPair>(impl::EntitySidePair(localElement, data.m_nearElementSide),
                            impl::ProcFaceIdPair(data.m_farElementProc, 0)));
        }
    }

    m_sideIdPool.generate_additional_ids_collective(m_graph.get_num_edges() + shellConnectivityList.size());
    fill_parallel_graph(shellNeighborsToReconnect);

    update_number_of_parallel_edges();
}

void ElemElemGraph::delete_elements(const stk::mesh::impl::DeletedElementInfoVector &elements_to_delete)
{
    std::vector<impl::ShellConnectivityData> shellConnectivityList;
    std::vector<impl::ElementSidePair> deletedShells;
    collect_local_shell_connectivity_data(elements_to_delete, shellConnectivityList, deletedShells);

    communicate_shell_connectivity(shellConnectivityList, deletedShells);

    std::vector<impl::DeletedElementData> local_elem_and_remote_connected_elem;
    delete_local_connections_and_collect_remote(elements_to_delete, local_elem_and_remote_connected_elem);

    std::vector< std::pair< stk::mesh::EntityId, stk::mesh::EntityId > > remote_edges;
    communicate_remote_connections_to_delete(local_elem_and_remote_connected_elem, remote_edges);

    clear_deleted_element_connections(elements_to_delete);

    delete_remote_connections(remote_edges);

    reconnect_volume_elements_across_deleted_shells(shellConnectivityList);
}

stk::mesh::ConnectivityOrdinal ElemElemGraph::get_neighboring_side_ordinal(const stk::mesh::BulkData &mesh,
                                                                           stk::mesh::Entity currentElem,
                                                                           stk::mesh::ConnectivityOrdinal currentOrdinal,
                                                                           stk::mesh::Entity neighborElem)
{
    stk::topology currentElemTopology = mesh.bucket(currentElem).topology();
    stk::topology currentSideTopology = currentElemTopology.side_topology(currentOrdinal);
    const stk::mesh::Entity* currentElemNodes = mesh.begin_nodes(currentElem);
    stk::mesh::EntityVector currentElemSideNodes(currentSideTopology.num_nodes());
    currentElemTopology.side_nodes(currentElemNodes, currentOrdinal, currentElemSideNodes.begin());

    stk::topology neighborTopology = mesh.bucket(neighborElem).topology();
    stk::mesh::EntityVector neighborSideNodes;

    bool foundNeighborOrdinal = false;
    unsigned neighborOrdinal = 0;
    for (; neighborOrdinal < neighborTopology.num_sides(); ++neighborOrdinal)
    {
        stk::topology neighborFaceTopology = neighborTopology.side_topology(neighborOrdinal);
        neighborSideNodes.resize(neighborFaceTopology.num_nodes());
        const stk::mesh::Entity* neighborNodes = mesh.begin_nodes(neighborElem);
        neighborTopology.side_nodes(neighborNodes, neighborOrdinal, neighborSideNodes.begin());
        std::pair<bool,unsigned> result = neighborFaceTopology.equivalent(currentElemSideNodes, neighborSideNodes);

        if (result.first && result.second >= neighborFaceTopology.num_positive_permutations())
        {
            foundNeighborOrdinal = true;
            break;
        }
    }
    ThrowRequireMsg(foundNeighborOrdinal, "Error: neighborElem is not a true neighbor of currentElem.");
    return static_cast<stk::mesh::ConnectivityOrdinal>(neighborOrdinal);
}

size_t ElemElemGraph::find_max_local_offset_in_neighborhood(stk::mesh::Entity element)
{
    stk::mesh::EntityVector side_nodes;
    stk::mesh::EntityVector connected_elements;
    stk::mesh::Bucket &elem_bucket = m_bulk_data.bucket(element);
    stk::topology topology = elem_bucket.topology();
    size_t num_sides = topology.num_sides();
    const stk::mesh::Entity* elem_nodes = elem_bucket.begin_nodes(m_bulk_data.bucket_ordinal(element));
    size_t max_local_offset = 0;
    size_t current_offset = element.local_offset();
    if (current_offset > max_local_offset)
    {
        max_local_offset = current_offset;
    }
    for (size_t side_index = 0; side_index < num_sides; ++side_index)
    {
        unsigned num_side_nodes = topology.side_topology(side_index).num_nodes();
        side_nodes.resize(num_side_nodes);
        topology.side_nodes(elem_nodes, side_index, side_nodes.begin());
        connected_elements.clear();
        impl::ConnectedElementDataVector connectedElementDataVector;
        impl::find_locally_owned_elements_these_nodes_have_in_common(m_bulk_data, num_side_nodes, side_nodes.data(), connected_elements);

        for (stk::mesh::Entity & connected_element : connected_elements)
        {
            current_offset = connected_element.local_offset();
            if (current_offset > max_local_offset)
            {
                max_local_offset = current_offset;
            }
        }
    }
    return max_local_offset;
}

void ElemElemGraph::resize_entity_to_local_id_vector_for_new_elements(const stk::mesh::EntityVector& allElementsNotAlreadyInGraph)
{
    size_t max_offset = 0;
    for (const stk::mesh::Entity& element_to_add : allElementsNotAlreadyInGraph)
    {
        const size_t local_max = find_max_local_offset_in_neighborhood(element_to_add);
        max_offset = std::max(local_max, max_offset);
    }
    resize_entity_to_local_id_if_needed(max_offset);
}

void ElemElemGraph::add_both_edges_between_local_elements(const GraphEdge& graphEdge)
{
    m_graph.add_edge(graphEdge);
    m_graph.add_edge(create_symmetric_edge(graphEdge));
}

void ElemElemGraph::add_local_edges(stk::mesh::Entity elem_to_add, impl::LocalId new_elem_id)
{
    std::vector<stk::mesh::GraphEdge> newGraphEdges;
    std::set<EntityId> localElementsConnectedToNewShell;
    get_element_side_pairs(m_bulk_data.mesh_index(elem_to_add), new_elem_id, newGraphEdges);
    for (const stk::mesh::GraphEdge& graphEdge : newGraphEdges)
    {
        stk::mesh::Entity neighbor = m_local_id_to_element_entity[graphEdge.elem2];
        if (is_valid_graph_element(neighbor))
        {
            add_both_edges_between_local_elements(graphEdge);
            if (m_element_topologies[new_elem_id].is_shell())
            {
                impl::LocalId neighbor_id2 = m_entity_to_local_id[neighbor.local_offset()];
                localElementsConnectedToNewShell.insert(neighbor_id2);
            }
        }
    }
}

void ElemElemGraph::add_vertex(impl::LocalId new_elem_id, stk::mesh::Entity elem_to_add)
{
    m_local_id_to_element_entity[new_elem_id] = elem_to_add;
    m_entity_to_local_id[elem_to_add.local_offset()] = new_elem_id;
    stk::topology elem_topology = m_bulk_data.bucket(elem_to_add).topology();
    m_element_topologies[new_elem_id] = elem_topology;
}

stk::mesh::EntityVector ElemElemGraph::filter_add_elements_arguments(const stk::mesh::EntityVector& allUnfilteredElementsNotAlreadyInGraph) const
{
    stk::mesh::EntityVector allElementsNotAlreadyInGraph;
    allElementsNotAlreadyInGraph.reserve(allUnfilteredElementsNotAlreadyInGraph.size());
    for(stk::mesh::Entity element : allUnfilteredElementsNotAlreadyInGraph)
    {
        ThrowRequire(m_bulk_data.is_valid(element));
        if(m_bulk_data.bucket(element).owned())
        {
            ThrowRequire(!is_valid_graph_element(element));
            allElementsNotAlreadyInGraph.push_back(element);
        }
    }
    return allElementsNotAlreadyInGraph;
}

void ElemElemGraph::add_elements_locally(const stk::mesh::EntityVector& allElementsNotAlreadyInGraph)
{
    resize_entity_to_local_id_vector_for_new_elements(allElementsNotAlreadyInGraph);
    for(stk::mesh::Entity newElem : allElementsNotAlreadyInGraph)
    {
        impl::LocalId newElemLocalId = get_new_local_element_id_from_pool();
        add_vertex(newElemLocalId, newElem);
        add_local_edges(newElem, newElemLocalId);
    }
}

void ElemElemGraph::add_elements(const stk::mesh::EntityVector &allUnfilteredElementsNotAlreadyInGraph)
{
    stk::mesh::EntityVector allElementsNotAlreadyInGraph = filter_add_elements_arguments(allUnfilteredElementsNotAlreadyInGraph);

    const size_t numEdgesBefore = num_edges();
    add_elements_locally(allElementsNotAlreadyInGraph);
    const size_t numNewSideIdsNeeded = num_edges() - numEdgesBefore;

    impl::ElemSideToProcAndFaceId elem_side_comm = get_element_side_ids_to_communicate();

    ssize_t num_additional_parallel_edges = elem_side_comm.size() - m_num_parallel_edges;
    if(num_additional_parallel_edges < 0)
    {
        num_additional_parallel_edges = 0;
    }
    size_t num_additional_side_ids_needed =  num_additional_parallel_edges + numNewSideIdsNeeded;
    m_sideIdPool.generate_additional_ids_collective(num_additional_side_ids_needed);

    stk::mesh::EntityVector allElementsNotAlreadyInGraph_copy = allElementsNotAlreadyInGraph;
    std::sort(allElementsNotAlreadyInGraph_copy.begin(), allElementsNotAlreadyInGraph_copy.end());

    std::set< stk::mesh::Entity > addedShells;
    impl::ElemSideToProcAndFaceId only_added_elements;
    for(impl::ElemSideToProcAndFaceId::value_type &elemSideToProcAndFaceId : elem_side_comm)
    {
        stk::mesh::Entity element = elemSideToProcAndFaceId.first.entity;
        stk::mesh::EntityVector::iterator elem_iter = std::lower_bound(allElementsNotAlreadyInGraph_copy.begin(), allElementsNotAlreadyInGraph_copy.end(), element);
        if(elem_iter!=allElementsNotAlreadyInGraph_copy.end() && *elem_iter==element)
        {
            only_added_elements.insert(elemSideToProcAndFaceId);
            if (m_bulk_data.bucket(element).topology().is_shell())
            {
                addedShells.insert(element);
            }
        }
    }

    fill_parallel_graph(only_added_elements);

    extract_coincident_edges_and_fix_chosen_side_ids_for_specified_elems(allElementsNotAlreadyInGraph);

    GraphInfo graphInfo(m_graph, m_parallelInfoForGraphEdges, m_element_topologies);
    remove_graph_edges_blocked_by_shell(graphInfo);

    stk::mesh::EntityVector addedShellsVector;
    for (auto &shell : addedShells)
    {
        addedShellsVector.push_back(shell);
    }

    stk::CommSparse comm(m_bulk_data.parallel());
    for (int phase = 0; phase < 2; ++phase)
    {
        pack_remote_edge_across_shell(comm, addedShellsVector, phase);
        if (0 == phase)
        {
            comm.allocate_buffers();
        }
        if (1 == phase)
        {
            comm.communicate();
        }
    }

    unpack_remote_edge_across_shell(comm);
    update_number_of_parallel_edges();
}

void ElemElemGraph::break_remote_shell_connectivity_and_pack(stk::CommSparse &comm, impl::LocalId leftId, impl::LocalId rightId, int phase)
{
    size_t index = 0;
    size_t numConnected = m_graph.get_num_edges_for_element(leftId);
    while(index < numConnected)
    {
        const GraphEdge & graphEdge = m_graph.get_edge_for_element(leftId, index);
        if(graphEdge.elem2 == rightId)
        {
            stk::mesh::Entity localElem = m_local_id_to_element_entity[leftId];
            int sharingProc = get_owning_proc_id_of_remote_element(localElem, index);
            if (phase == 1)
            {
                m_graph.delete_edge_from_graph(leftId, index);
            }
            else
            {
                ++index;
            }

            stk::mesh::EntityId localElemId = m_bulk_data.identifier(localElem);

            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(localElemId);
            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(-rightId);
            break;
        }
        else
        {
            ++index;
        }
    }
}

void ElemElemGraph::pack_both_remote_shell_connectivity(stk::CommSparse &comm, impl::LocalId shellId, impl::LocalId leftId, impl::LocalId rightId)
{
    size_t index = 0;
    size_t num_connected = m_graph.get_num_edges_for_element(shellId);
    while(index < num_connected)
    {
        const GraphEdge & graphEdge = m_graph.get_edge_for_element(shellId, index);
        if (graphEdge.elem2 == rightId)
        {
            stk::mesh::Entity shell = m_local_id_to_element_entity[shellId];
            int sharingProc = get_owning_proc_id_of_remote_element(shell, index);

            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(-leftId);
            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(-rightId);
        }
        else if (graphEdge.elem2 == leftId)
        {
            stk::mesh::Entity shell = m_local_id_to_element_entity[shellId];
            int sharingProc = get_owning_proc_id_of_remote_element(shell, index);

            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(-rightId);
            comm.send_buffer(sharingProc).pack<stk::mesh::EntityId>(-leftId);
        }
        ++index;
    }
}

void ElemElemGraph::pack_remote_edge_across_shell(stk::CommSparse &comm, stk::mesh::EntityVector &addedShells, int phase)
{
    for(stk::mesh::Entity &shell : addedShells)
    {
        impl::LocalId shellId = m_entity_to_local_id[shell.local_offset()];
        impl::LocalId leftId = impl::INVALID_LOCAL_ID;
        impl::LocalId rightId = impl::INVALID_LOCAL_ID;
        size_t numConnected = m_graph.get_num_edges_for_element(shellId);
        for(size_t i=0; i<numConnected; i++)
        {
            const GraphEdge & graphEdge = m_graph.get_edge_for_element(shellId, i);
            if (leftId == impl::INVALID_LOCAL_ID)
            {
                leftId = graphEdge.elem2;
                continue;
            }
            if (rightId == impl::INVALID_LOCAL_ID)
            {
                rightId = graphEdge.elem2;
                continue;
            }
        }
        bool isLeftRemote = leftId < 0;
        bool isRightRemote = rightId < 0;

        if (leftId == impl::INVALID_LOCAL_ID || rightId == impl::INVALID_LOCAL_ID)
        {
            continue;
        }

        if (!isLeftRemote && !isRightRemote)
        {
            continue;
        }

        if (isLeftRemote && isRightRemote) {
            pack_both_remote_shell_connectivity(comm, shellId, leftId, rightId);
        }
        else if (!isLeftRemote) {
            break_remote_shell_connectivity_and_pack(comm, leftId, rightId, phase);
        }
        else if (!isRightRemote) {
            break_remote_shell_connectivity_and_pack(comm, rightId, leftId, phase);
        }
    }
}

void ElemElemGraph::unpack_remote_edge_across_shell(stk::CommSparse &comm)
{
    for(int i=0;i<m_bulk_data.parallel_size();++i)
    {
        while(comm.recv_buffer(i).remaining())
        {
            stk::mesh::EntityId localElemId;
            stk::mesh::EntityId remoteElemId;
            comm.recv_buffer(i).unpack<stk::mesh::EntityId>(remoteElemId);
            comm.recv_buffer(i).unpack<stk::mesh::EntityId>(localElemId);
            stk::mesh::Entity localElem = m_bulk_data.get_entity(stk::topology::ELEM_RANK, localElemId);
            impl::LocalId localId = m_entity_to_local_id[localElem.local_offset()];

            size_t index = 0;
            size_t numConnected = m_graph.get_num_edges_for_element(localId);
            while(index <numConnected)
            {
                const GraphEdge & graphEdge = m_graph.get_edge_for_element(localId, index);
                if(graphEdge.elem2 == static_cast<impl::LocalId>(-1*remoteElemId))
                {
                    m_graph.delete_edge_from_graph(localId, index);
                    break;
                }
                else
                {
                    ++index;
                }
            }
        }
    }
}

void add_downward_connected_from_elements(stk::mesh::BulkData &bulkData,
                                          std::vector<std::pair<stk::mesh::Entity, int> > &elem_proc_pairs_to_move,
                                          std::vector<std::pair<stk::mesh::Entity, int> > &entity_proc_pairs_to_move)
{
    for(stk::mesh::EntityRank rank=stk::topology::NODE_RANK; rank<stk::topology::ELEMENT_RANK; rank++)
    {
        for(size_t i=0; i<elem_proc_pairs_to_move.size(); i++)
        {
            entity_proc_pairs_to_move.push_back(elem_proc_pairs_to_move[i]);
            stk::mesh::Entity elem = elem_proc_pairs_to_move[i].first;
            int otherProc = elem_proc_pairs_to_move[i].second;
            const unsigned numConnected = bulkData.num_connectivity(elem, rank);
            const stk::mesh::Entity *connectedEntity = bulkData.begin(elem, rank);
            for(unsigned j=0; j<numConnected; j++)
            {
                if(bulkData.bucket(connectedEntity[j]).owned())
                {
                    entity_proc_pairs_to_move.push_back(std::make_pair(connectedEntity[j], otherProc));
                }
            }
        }
    }
}

impl::parallel_info ElemElemGraph::create_parallel_info(stk::mesh::Entity connected_element,
                                                        const stk::mesh::Entity elemToSend,
                                                        int elemToSendSideId,
                                                        const int destination_proc)
{
    stk::mesh::Permutation permutation = get_permutation_given_neighbors_node_ordering(connected_element, elemToSend, elemToSendSideId);
    stk::mesh::EntityId faceId = m_sideIdPool.get_available_id();
    stk::topology elemTopology = m_bulk_data.bucket(elemToSend).topology();
    bool inBodyToBeSkinned = m_skinned_selector(m_bulk_data.bucket(connected_element));
    return impl::parallel_info(destination_proc, permutation, faceId, elemTopology, inBodyToBeSkinned);
}

stk::mesh::Permutation ElemElemGraph::get_permutation_given_neighbors_node_ordering(stk::mesh::Entity neighborElem,
                                                                                    const stk::mesh::Entity elemToSend,
                                                                                    int elemToSendSideId)
{
    stk::topology elemTopology = m_bulk_data.bucket(elemToSend).topology();
    const stk::mesh::Entity *elemNodes = m_bulk_data.begin_nodes(elemToSend);

    stk::topology sideTopology = elemTopology.side_topology(elemToSendSideId);
    std::vector<stk::mesh::Entity> sideNodes(sideTopology.num_nodes());

    elemTopology.side_nodes(elemNodes, elemToSendSideId, sideNodes.begin());

    stk::mesh::EntityRank sideRank = m_bulk_data.mesh_meta_data().side_rank();
    stk::mesh::OrdinalAndPermutation ordperm = get_ordinal_and_permutation(m_bulk_data, neighborElem, sideRank, sideNodes);

    return ordperm.second;
}



void ElemElemGraph::update_all_local_neighbors(const stk::mesh::Entity elemToSend,
                                               const int destinationProc,
                                               impl::ParallelGraphInfo &newParallelGraphEntries)
{
    stk::mesh::EntityId elemGlobalId = m_bulk_data.identifier(elemToSend);
    size_t numNeighbors = get_num_connected_elems(elemToSend);

    for(size_t k = 0; k < numNeighbors; k++)
    {
        if(is_connected_elem_locally_owned(elemToSend, k))
        {
            impl::ElementViaSidePair neighborElemViaSide = get_connected_element_and_via_side(elemToSend, k);
            stk::mesh::Entity neighborElem = neighborElemViaSide.element;
            int elemToSendSideId = neighborElemViaSide.side;

            //impl::LocalId localId = get_local_element_id(neighborElem);
            impl::parallel_info parallelInfo = create_parallel_info(neighborElem, elemToSend, elemToSendSideId, destinationProc);
            impl::LocalId elemToSendId = get_local_element_id(elemToSend);
            const GraphEdge& graphEdge = m_graph.get_edge_for_element(elemToSendId, k);
            GraphEdge symmGraphEdge(graphEdge.elem2, graphEdge.side2, -elemGlobalId, graphEdge.side1);
            newParallelGraphEntries.insert(std::make_pair(symmGraphEdge, parallelInfo));
        }
    }
}


void ElemElemGraph::create_parallel_graph_info_needed_once_entities_are_moved(
        const stk::mesh::EntityProcVec &elemProcPairsToMove,
        impl::ParallelGraphInfo &newParallelGraphEntries)
{
    for(size_t i = 0; i < elemProcPairsToMove.size(); i++)
    {
        stk::mesh::Entity elemToSend = elemProcPairsToMove[i].first;
        int destination_proc = elemProcPairsToMove[i].second;

        update_all_local_neighbors(elemToSend, destination_proc, newParallelGraphEntries);
    }
}

stk::mesh::EntityId ElemElemGraph::get_available_side_id()
{
    return m_sideIdPool.get_available_id();
}

stk::mesh::Entity ElemElemGraph::add_side_to_mesh(const stk::mesh::impl::ElementSidePair& sidePair, const stk::mesh::PartVector& skinParts, stk::mesh::EntityId sideId)
{
    stk::mesh::Entity element = m_local_id_to_element_entity[sidePair.first];
    int side_ordinal = sidePair.second;
    stk::mesh::Entity side = stk::mesh::impl::get_side_for_element(m_bulk_data, element, side_ordinal);
    if(m_bulk_data.is_valid(side))
    {
        if(m_bulk_data.bucket(side).owned())
        {
            m_bulk_data.change_entity_parts(side, skinParts, stk::mesh::PartVector());
        }
    }
    else
    {
        stk::mesh::PartVector add_parts = skinParts;
        stk::topology elem_top = m_bulk_data.bucket(element).topology();
        stk::topology side_top = elem_top.side_topology(side_ordinal);
        add_parts.push_back(&m_bulk_data.mesh_meta_data().get_topology_root_part(side_top));
        ThrowRequireMsg(!impl::is_id_already_in_use_locally(m_bulk_data, m_bulk_data.mesh_meta_data().side_rank(), sideId), "Program error. Id in use.");
        side = stk::mesh::declare_element_side(m_bulk_data, sideId, element, side_ordinal, add_parts);
    }
    return side;
}

void add_skinned_shared_side_to_element(stk::mesh::BulkData& bulkData, const stk::mesh::GraphEdge& graphEdge, const impl::parallel_info& parallel_edge_info,
        stk::mesh::Entity local_element, const stk::mesh::PartVector& parts_for_creating_side,
        std::vector<stk::mesh::sharing_info> &shared_modified, const stk::mesh::PartVector *boundary_mesh_parts = nullptr)
{
    int side_id = graphEdge.side1;
    ThrowRequireMsg(side_id != -1, "Program error. Please contact sierra-help@sandia.gov for support.");

    stk::mesh::EntityId side_global_id = parallel_edge_info.m_chosen_side_id;
    stk::mesh::ConnectivityOrdinal side_ord = static_cast<stk::mesh::ConnectivityOrdinal>(side_id);
    std::string msg = "Program error. Contact sierra-help@sandia.gov for support.";

    // determine which element is active
    stk::mesh::Permutation perm = stk::mesh::DEFAULT_PERMUTATION;
    int other_proc = parallel_edge_info.m_other_proc;
    int owning_proc = std::min(other_proc, bulkData.parallel_rank());

    if(parallel_edge_info.m_in_body_to_be_skinned)
    {
        perm = static_cast<stk::mesh::Permutation>(parallel_edge_info.m_permutation);
    }

    stk::mesh::Entity side = stk::mesh::impl::get_side_for_element(bulkData, local_element, side_id);

    if(!bulkData.is_valid(side))
    {
        ThrowRequireMsg(!impl::is_id_already_in_use_locally(bulkData, bulkData.mesh_meta_data().side_rank(), side_global_id), msg);
        stk::mesh::PartVector add_parts = parts_for_creating_side;
        stk::topology elem_top = bulkData.bucket(local_element).topology();
        stk::topology side_top = elem_top.side_topology(side_ord);
        add_parts.push_back(&bulkData.mesh_meta_data().get_topology_root_part(side_top));
        side = impl::connect_side_to_element(bulkData, local_element, side_global_id, side_ord, perm, add_parts);
        shared_modified.push_back(stk::mesh::sharing_info(side, other_proc, owning_proc));
    }
    else
    {
        if(bulkData.bucket(side).owned())
        {
            bulkData.change_entity_parts(side, parts_for_creating_side, stk::mesh::PartVector());
        }
        shared_modified.push_back(stk::mesh::sharing_info(side, other_proc, bulkData.parallel_owner_rank(side)));
    }
}

struct LocalEdge
{
    stk::mesh::Entity m_element;
    stk::mesh::Entity m_other_element;
    LocalEdge(stk::mesh::Entity element, stk::mesh::Entity other_element) :
        m_element(element), m_other_element(other_element)
    {}

};

struct RemoteEdge
{
    const GraphEdge& m_graphEdge;
    stk::mesh::impl::parallel_info m_parallel_edge_info;
    RemoteEdge(const GraphEdge& graphEdge, stk::mesh::impl::parallel_info parallel_edge_info) :
        m_graphEdge(graphEdge), m_parallel_edge_info(parallel_edge_info)
    {}

};

void ElemElemGraph::create_remote_sides(stk::mesh::BulkData& bulk_data, const std::vector<RemoteEdge>& remote_edges, stk::mesh::EntityVector& skinned_elements, const stk::mesh::PartVector& skin_parts,
        const std::vector<unsigned>& side_counts, std::vector<stk::mesh::sharing_info>& shared_modified)
{
    for(size_t k=0;k<remote_edges.size();++k)
    {
        int side_ordinal = remote_edges[k].m_graphEdge.side1;
        stk::mesh::Entity element = m_local_id_to_element_entity[remote_edges[k].m_graphEdge.elem1];
        unsigned num_connected_elements_this_side = get_num_connected_elems(element, side_ordinal);
        if(num_connected_elements_this_side == side_counts[side_ordinal])
        {
            skinned_elements.push_back(element);
            add_skinned_shared_side_to_element(bulk_data, remote_edges[k].m_graphEdge, remote_edges[k].m_parallel_edge_info, element, skin_parts, shared_modified);
        }
    }
}

void ElemElemGraph::create_remote_sides1(stk::mesh::BulkData& bulk_data, const std::vector<RemoteEdge>& remote_edges, stk::mesh::EntityVector& skinned_elements, const stk::mesh::PartVector& skin_parts,
        const std::vector<unsigned>& side_counts, std::vector<stk::mesh::sharing_info>& shared_modified)
{
    for(size_t k=0;k<remote_edges.size();++k)
    {
        int side_ordinal = remote_edges[k].m_graphEdge.side1;
        if(side_counts[side_ordinal] == 1)
        {
            stk::mesh::Entity element = m_local_id_to_element_entity[remote_edges[k].m_graphEdge.elem1];
            skinned_elements.push_back(element);
            add_skinned_shared_side_to_element(bulk_data, remote_edges[k].m_graphEdge, remote_edges[k].m_parallel_edge_info, element, skin_parts, shared_modified);
        }
    }
}

bool ElemElemGraph::is_element_selected_and_can_have_side(const stk::mesh::BulkData &bulkData, const stk::mesh::Selector &selector, stk::mesh::Entity otherElement)
{
    return selector(bulkData.bucket(otherElement)) && impl::does_element_have_side(bulkData, otherElement);
}

bool is_remote_element_air(const ParallelInfoForGraphEdges &parallelInfoForGraphEdges, const stk::mesh::GraphEdge &graphEdge)
{
    const stk::mesh::impl::parallel_info &parInfo = parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
    return parInfo.m_is_air;
}

bool ElemElemGraph::is_connected_element_air(const stk::mesh::GraphEdge &graphEdge)
{
    if(impl::is_local_element(graphEdge.elem2))
        return is_element_selected_and_can_have_side(m_bulk_data, *m_air_selector, m_local_id_to_element_entity[graphEdge.elem2]);
    else
        return is_remote_element_air(m_parallelInfoForGraphEdges, graphEdge);
}

bool is_remote_element_in_body_to_be_skinned(const ParallelInfoForGraphEdges &parallelInfoForGraphEdges, const stk::mesh::GraphEdge &graphEdge)
{
    const stk::mesh::impl::parallel_info &parInfo = parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
    return parInfo.m_in_body_to_be_skinned;
}

bool ElemElemGraph::is_connected_element_in_body_to_be_skinned(const stk::mesh::GraphEdge &graphEdge)
{
    return (!impl::is_local_element(graphEdge.elem2) && is_remote_element_in_body_to_be_skinned(m_parallelInfoForGraphEdges, graphEdge));
}

void ElemElemGraph::add_exposed_sides_due_to_air_selector(impl::LocalId local_id, std::vector<int> &exposedSides)
{
    std::vector<int> exposedSidesDueToAirSelector;
    std::vector<bool> areAllConnectedElemsAir(m_element_topologies[local_id].num_sides(), true);
    for(size_t k = 0; k < m_graph.get_num_edges_for_element(local_id); ++k)
    {
        const GraphEdge & graphEdge = m_graph.get_edge_for_element(local_id, k);
        if(is_connected_element_air(graphEdge))
            exposedSidesDueToAirSelector.push_back(graphEdge.side1);
        else
            areAllConnectedElemsAir[graphEdge.side1] = false;
    }

    for(int exposedSide : exposedSidesDueToAirSelector)
        if(areAllConnectedElemsAir[exposedSide])
            exposedSides.push_back(exposedSide);
}

std::vector<int> ElemElemGraph::get_sides_exposed_on_other_procs(stk::mesh::impl::LocalId localId, int numElemSides)
{
    std::vector<bool> isConnectedToRemoteElementInBodyToSkin(numElemSides, false);
    std::vector<bool> isOnlyConnectedRemotely(numElemSides, true);
    for(const stk::mesh::GraphEdge &graphEdge : m_graph.get_edges_for_element(localId))
    {
        if(impl::is_local_element(graphEdge.elem2))
        {
            stk::mesh::Entity other_element = m_local_id_to_element_entity[graphEdge.elem2];
            if(is_element_selected_and_can_have_side(m_bulk_data, m_skinned_selector, other_element))
                isOnlyConnectedRemotely[graphEdge.side1] = false;
        }
        else
        {
            impl::parallel_info &parallel_edge_info = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
            bool is_other_element_selected = parallel_edge_info.m_in_body_to_be_skinned;
            if(is_other_element_selected)
                isConnectedToRemoteElementInBodyToSkin[graphEdge.side1] = true;
        }
    }

    std::vector<int> exposedSides;
    for(int side = 0; side < numElemSides; side++)
    {
        if(isConnectedToRemoteElementInBodyToSkin[side] && isOnlyConnectedRemotely[side])
        {
            exposedSides.push_back(side);
        }
    }
    return exposedSides;
}

std::vector<SideSetEntry> ElemElemGraph::extract_skinned_sideset(  )
{
    std::vector<SideSetEntry> skinnedSideSet;

    const stk::mesh::BucketVector& buckets = m_bulk_data.get_buckets(stk::topology::ELEM_RANK, m_bulk_data.mesh_meta_data().locally_owned_part());

    for(size_t i=0;i<buckets.size();++i)
    {
        const stk::mesh::Bucket &bucket = *buckets[i];
        for(size_t j=0;j<bucket.size();++j)
        {
            stk::mesh::Entity element = bucket[j];

            stk::mesh::impl::LocalId localId = get_local_element_id(element);
            std::vector<int> exposedSides = get_sides_for_skinning(bucket, element, localId);

            for(size_t k=0; k<exposedSides.size(); ++k)
            {
                skinnedSideSet.push_back(SideSetEntry(element, static_cast<ConnectivityOrdinal> (exposedSides[k])));
            }
        }
    }

    return skinnedSideSet;
}

std::vector<int> ElemElemGraph::get_exposed_sides(stk::mesh::impl::LocalId localId, int numElemSides)
{
    std::vector<int> exposedSides;
    impl::add_exposed_sides(localId, numElemSides, m_graph, exposedSides);
    if(m_air_selector != nullptr)
        add_exposed_sides_due_to_air_selector(localId, exposedSides);
    return exposedSides;
}

std::vector<int> ElemElemGraph::get_sides_for_skinning(const stk::mesh::Bucket& bucket,
                                                       stk::mesh::Entity element,
                                                       stk::mesh::impl::LocalId localId)
{
    int numElemSides = m_element_topologies[localId].num_sides();
    std::vector<int> exposedSides;
    if(impl::does_element_have_side(m_bulk_data, element))
    {
        if(m_skinned_selector(bucket))
            exposedSides = get_exposed_sides(localId, numElemSides);
        else if(m_air_selector != nullptr && (*m_air_selector)(bucket))
            exposedSides = get_sides_exposed_on_other_procs(localId, numElemSides);
    }
    return exposedSides;
}

void ElemElemGraph::skin_mesh(const stk::mesh::PartVector& skinParts)
{
    const stk::mesh::BucketVector& buckets = m_bulk_data.get_buckets(stk::topology::ELEM_RANK, m_bulk_data.mesh_meta_data().locally_owned_part());

    std::vector<stk::mesh::sharing_info> sharedModified;

    m_bulk_data.modification_begin();
    for(size_t i=0;i<buckets.size();++i)
    {
        const stk::mesh::Bucket &bucket = *buckets[i];
        for(size_t j=0;j<bucket.size();++j)
        {
            stk::mesh::Entity element = bucket[j];

            stk::mesh::impl::LocalId localId = get_local_element_id(element);
            std::vector<int> exposedSides = get_sides_for_skinning(bucket, element, localId);
            stk::util::sort_and_unique(exposedSides);
            create_side_entities(exposedSides, localId, skinParts, sharedModified);
        }
    }
    m_bulk_data.make_mesh_parallel_consistent_after_skinning(sharedModified);
}

void ElemElemGraph::create_side_entities(const std::vector<int> &exposedSides,
                                         impl::LocalId localId,
                                         const stk::mesh::PartVector& skinParts,
                                         std::vector<stk::mesh::sharing_info> &sharedModified)
{
    stk::mesh::SideConnector sideConnector = get_side_connector();
    stk::mesh::Entity element = m_local_id_to_element_entity[localId];
    for(size_t i=0;i<exposedSides.size();++i)
    {
        stk::mesh::EntityId newFaceId = 0;
        for(const GraphEdge & graphEdge : m_graph.get_edges_for_element(localId))
        {
            newFaceId = add_side_for_remote_edge(graphEdge, exposedSides[i], element, skinParts, sharedModified);
            if(newFaceId != 0)
                break;
        }
        auto iter = m_coincidentGraph.find(localId);
        if(iter != m_coincidentGraph.end())
        {
            for(const GraphEdge & graphEdge : iter->second)
            {
                newFaceId = add_side_for_remote_edge(graphEdge, exposedSides[i], element, skinParts, sharedModified);
                if(newFaceId != 0)
                    break;
            }
        }
        if(newFaceId == 0)
        {
            newFaceId = m_sideIdPool.get_available_id();
        }

        stk::mesh::Entity sideEntity = add_side_to_mesh({localId, exposedSides[i]}, skinParts, newFaceId);

        sideConnector.connect_side_to_all_elements(sideEntity, element, exposedSides[i]);
    }
}

stk::mesh::EntityId ElemElemGraph::add_side_for_remote_edge(const GraphEdge & graphEdge,
                                                            int elemSide,
                                                            stk::mesh::Entity element,
                                                            const stk::mesh::PartVector& skin_parts,
                                                            std::vector<stk::mesh::sharing_info> &shared_modified)
{
    stk::mesh::EntityId newFaceId = 0;
    if(graphEdge.side1 == elemSide)
    {
        if(!impl::is_local_element(graphEdge.elem2))
        {
            impl::parallel_info &parallel_edge_info = m_parallelInfoForGraphEdges.get_parallel_info_for_graph_edge(graphEdge);
            newFaceId = parallel_edge_info.m_chosen_side_id;
            add_skinned_shared_side_to_element(m_bulk_data, graphEdge, parallel_edge_info, element, skin_parts, shared_modified);
        }
    }
    return newFaceId;
}

}} // end namespaces stk mesh

