// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_REPARTITIONFACTORY_DEF_HPP
#define MUELU_REPARTITIONFACTORY_DEF_HPP

#include <algorithm>
#include <iostream>
#include <sstream>

#include "MueLu_RepartitionFactory_decl.hpp" // TMP JG NOTE: before other includes, otherwise I cannot test the fwd declaration in _def

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_TabularOutputter.hpp>

#include <Xpetra_Export.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_VectorFactory.hpp>

#ifdef HAVE_MUELU_ZOLTAN2
#include <Zoltan2_TaskMapping.hpp>
#endif

#include "MueLu_Utilities.hpp"

#include "MueLu_Level.hpp"
#include "MueLu_MasterList.hpp"
#include "MueLu_Monitor.hpp"

#ifdef HAVE_MUELU_RCA
extern "C" {
#include <rca_lib.h>
}
#endif

namespace MueLu {

  static int machineDims[3] = { 17, 8, 24 };

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

#define SET_VALID_ENTRY(name) validParamList->setEntry(name, MasterList::getEntry(name))
    SET_VALID_ENTRY("repartition: start level");
    SET_VALID_ENTRY("repartition: min rows per proc");
    SET_VALID_ENTRY("repartition: max imbalance");
    SET_VALID_ENTRY("repartition: print partition distribution");
    SET_VALID_ENTRY("repartition: remap num values");
    SET_VALID_ENTRY("repartition: remap algorithm");
    {
      typedef Teuchos::StringToIntegralParameterEntryValidator<int> validatorType;
      validParamList->getEntry("repartition: remap algorithm").setValidator(
        rcp(new validatorType(Teuchos::tuple<std::string>(
              "none",                           // No remapping (use results of the partitioner)
              "muelu",                          // Remapping using partial bipartite graph
              "muelu-hops",                     // Similar to "muelu", but edge weights include hop metrics
              "zoltan2_task_map"                // Remapping using task mapper (typically, K-means clustering)
            ), "repartition: remap algorithm")));
    }

    validParamList->set< RCP<const FactoryBase> >("A",              Teuchos::null, "Factory of the matrix A");
    validParamList->set< RCP<const FactoryBase> >("Partition",      Teuchos::null, "Factory of the partition");
    validParamList->set< RCP<const FactoryBase> >("Coordinates",    Teuchos::null, "Factory of the coordinates");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level &currentLevel) const {
    Input(currentLevel, "A");
    Input(currentLevel, "Partition");

    const Teuchos::ParameterList& pL = GetParameterList();
    if (pL.get<std::string>("repartition: remap algorithm") == "zoltan2_task_map")
      Input(currentLevel, "Coordinates");
  }

  template<class T> class MpiTypeTraits            { public: static MPI_Datatype getType(); };
  template<>        class MpiTypeTraits<long>      { public: static MPI_Datatype getType() { return MPI_LONG;      } };
  template<>        class MpiTypeTraits<int>       { public: static MPI_Datatype getType() { return MPI_INT;       } };
  template<>        class MpiTypeTraits<short>     { public: static MPI_Datatype getType() { return MPI_SHORT;     } };
#ifdef HAVE_TEUCHOS_LONG_LONG_INT
  template<>        class MpiTypeTraits<long long> { public: static MPI_Datatype getType() { return MPI_LONG_LONG; } };
#endif
  template<>        class MpiTypeTraits<double>    { public: static MPI_Datatype getType() { return MPI_DOUBLE;    } };

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
    FactoryMonitor m(*this, "Build", currentLevel);

    const Teuchos::ParameterList& pL = GetParameterList();
    // Access parameters here to make sure that we set the parameter entry flag to "used" even in case of short-circuit evaluation.
    // TODO (JG): I don't really know if we want to do this.
    const int    startLevel          = pL.get<int>   ("repartition: start level");
    const LO     minRowsPerProcessor = pL.get<LO>    ("repartition: min rows per proc");
    const double nonzeroImbalance    = pL.get<double>("repartition: max imbalance");

    // TODO: We only need a CrsGraph. This class does not have to be templated on Scalar types.
    RCP<Matrix> A = Get< RCP<Matrix> >(currentLevel, "A");

    // ======================================================================================================
    // Determine whether partitioning is needed
    // ======================================================================================================
    // NOTE: most tests include some global communication, which is why we currently only do tests until we make
    // a decision on whether to repartition. However, there is value in knowing how "close" we are to having to
    // rebalance an operator. So, it would probably be beneficial to do and report *all* tests.

    // Test1: skip repartitioning if current level is less than the specified minimum level for repartitioning
    if (currentLevel.GetLevelID() < startLevel) {
      GetOStream(Statistics0) << "Repartitioning?  NO:" <<
          "\n  current level = " << toString(currentLevel.GetLevelID()) <<
          ", first level where repartitioning can happen is " + toString(startLevel) << std::endl;

      Set<RCP<const Import> >(currentLevel, "Importer", Teuchos::null);
      return;
    }

    RCP<const Map> rowMap = A->getRowMap();

    // NOTE: Teuchos::MPIComm::duplicate() calls MPI_Bcast inside, so this is
    // a synchronization point. However, as we do sumAll afterwards anyway, it
    // does not matter.
    RCP<const Teuchos::Comm<int> > origComm = rowMap->getComm();
    RCP<const Teuchos::Comm<int> > comm     = origComm->duplicate();

    // Test 2: check whether A is actually distributed, i.e. more than one processor owns part of A
    // TODO: this global communication can be avoided if we store the information with the matrix (it is known when matrix is created)
    // TODO: further improvements could be achieved when we use subcommunicator for the active set. Then we only need to check its size
    {
      int numActiveProcesses = 0;
      sumAll(comm, Teuchos::as<int>((A->getNodeNumRows() > 0) ? 1 : 0), numActiveProcesses);

      if (numActiveProcesses == 1) {
        GetOStream(Statistics0) << "Repartitioning?  NO:" <<
            "\n  # processes with rows = " << toString(numActiveProcesses) << std::endl;

        Set<RCP<const Import> >(currentLevel, "Importer", Teuchos::null);
        return;
      }
    }

    bool test3 = false, test4 = false;
    std::string msg3, msg4;

    // Test3: check whether number of rows on any processor satisfies the minimum number of rows requirement
    // NOTE: Test2 ensures that repartitionning is not done when there is only one processor (it may or may not satisfy Test3)
    if (minRowsPerProcessor > 0) {
      LO numMyRows = Teuchos::as<LO>(A->getNodeNumRows()), minNumRows, LOMAX = Teuchos::OrdinalTraits<LO>::max();
      LO haveFewRows = (numMyRows < minRowsPerProcessor ? 1 : 0), numWithFewRows = 0;
      sumAll(comm, haveFewRows, numWithFewRows);
      minAll(comm, (numMyRows > 0 ? numMyRows : LOMAX), minNumRows);

      // TODO: we could change it to repartition only if the number of processors with numRows < minNumRows is larger than some
      // percentage of the total number. This way, we won't repartition if 2 out of 1000 processors don't have enough elements.
      // I'm thinking maybe 20% threshold. To implement, simply add " && numWithFewRows < .2*numProcs" to the if statement.
      if (numWithFewRows > 0)
        test3 = true;

      msg3 = "\n  min # rows per proc = " + toString(minNumRows) + ", min allowable = " + toString(minRowsPerProcessor);
    }

    // Test4: check whether the balance in the number of nonzeros per processor is greater than threshold
    if (!test3) {
      GO minNnz, maxNnz, numMyNnz = Teuchos::as<GO>(A->getNodeNumEntries());
      maxAll(comm, numMyNnz,                           maxNnz);
      minAll(comm, (numMyNnz > 0 ? numMyNnz : maxNnz), minNnz); // min nnz over all active processors
      double imbalance = Teuchos::as<double>(maxNnz)/minNnz;

      if (imbalance > nonzeroImbalance)
        test4 = true;

      msg4 = "\n  nonzero imbalance = " + toString(imbalance) + ", max allowable = " + toString(nonzeroImbalance);
    }

    if (!test3 && !test4) {
      GetOStream(Statistics0) << "Repartitioning?  NO:" << msg3 + msg4 << std::endl;

      Set<RCP<const Import> >(currentLevel, "Importer", Teuchos::null);
      return;
    }

    GetOStream(Statistics0) << "Repartitioning? YES:" << msg3 + msg4 << std::endl;

    GO                     indexBase = rowMap->getIndexBase();
    Xpetra::UnderlyingLib  lib       = rowMap->lib();
    int myRank   = comm->getRank();
    int numProcs = comm->getSize();

    RCP<const Teuchos::MpiComm<int> > tmpic = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
    TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, Exceptions::RuntimeError, "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
    RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();

    // ======================================================================================================
    // Calculate number of partitions
    // ======================================================================================================
    // FIXME Quick way to figure out how many partitions there should be (same algorithm as ML)
    // FIXME Should take into account nnz? Perhaps only when user is using min #nnz per row threshold.
    GO numPartitions;
    if (currentLevel.IsAvailable("number of partitions")) {
      numPartitions = currentLevel.Get<GO>("number of partitions");
      GetOStream(Warnings0) << "Using user-provided \"number of partitions\", the performance is unknown" << std::endl;

    } else {
      if (Teuchos::as<GO>(A->getGlobalNumRows()) < minRowsPerProcessor) {
        // System is too small, migrate it to a single processor
        numPartitions = 1;

      } else {
        // Make sure that each processor has approximately minRowsPerProcessor
        numPartitions = A->getGlobalNumRows() / minRowsPerProcessor;
      }
      numPartitions = std::min(numPartitions, Teuchos::as<GO>(numProcs));

      currentLevel.Set("number of partitions", numPartitions, NoFactory::get());
    }
    GetOStream(Statistics0) << "Number of partitions to use = " << numPartitions << std::endl;

    // ======================================================================================================
    // Construct decomposition vector
    // ======================================================================================================
    RCP<GOVector> decomposition;
    if (numPartitions == 1) {
      // Trivial case: decomposition is the trivial one, all zeros. We skip the call to Zoltan_Interface
      // (this is mostly done to avoid extra output messages, as even if we didn't skip there is a shortcut
      // in Zoltan[12]Interface).
      // TODO: We can probably skip more work in this case (like building all extra data structures)
      GetOStream(Warnings0) << "Only one partition: Skip call to the repartitioner." << std::endl;
      decomposition = Xpetra::VectorFactory<GO, LO, GO, NO>::Build(A->getRowMap(), true);

    } else {
      decomposition = Get<RCP<GOVector> >(currentLevel, "Partition");

      if (decomposition.is_null()) {
        GetOStream(Warnings0) << "No repartitioning necessary: partitions were left unchanged by the repartitioner" << std::endl;
        Set<RCP<const Import> >(currentLevel, "Importer", Teuchos::null);
        return;
      }
    }

    // ======================================================================================================
    // Remap if necessary
    // ======================================================================================================
    // From a user perspective, we want user to not care about remapping, thinking of it as only a performance feature.
    // There are two problems, however.
    // (1) Next level aggregation depends on the order of GIDs in the vector, if one uses "natural" or "random" orderings.
    //     This also means that remapping affects next level aggregation, despite the fact that the _set_ of GIDs for
    //     each partition is the same.
    // (2) Even with the fixed order of GIDs, the remapping may influence the aggregation for the next-next level.
    //     Let us consider the following example. Lets assume that when we don't do remapping, processor 0 would have
    //     GIDs {0,1,2}, and processor 1 GIDs {3,4,5}, and if we do remapping processor 0 would contain {3,4,5} and
    //     processor 1 {0,1,2}. Now, when we run repartitioning algorithm on the next level (say Zoltan1 RCB), it may
    //     be dependent on whether whether it is [{0,1,2}, {3,4,5}] or [{3,4,5}, {0,1,2}]. Specifically, the tie-breaking
    //     algorithm can resolve these differently. For instance, running
    //         mpirun -np 5 ./MueLu_ScalingTestParamList.exe --xml=easy_sa.xml --nx=12 --ny=12 --nz=12
    //     with
    //         <ParameterList name="MueLu">
    //           <Parameter name="coarse: max size"                type="int"      value="1"/>
    //           <Parameter name="repartition: enable"             type="bool"     value="true"/>
    //           <Parameter name="repartition: min rows per proc"  type="int"      value="2"/>
    //           <ParameterList name="level 1">
    //             <Parameter name="repartition: remap parts"      type="bool"     value="false/true"/>
    //           </ParameterList>
    //         </ParameterList>
    //     produces different repartitioning for level 2.
    //     This different repartitioning may then escalate into different aggregation for the next level.
    //
    // We fix (1) by fixing the order of GIDs in a vector by sorting the resulting vector.
    // Fixing (2) is more complicated.
    // FIXME: Fixing (2) in Zoltan may not be enough, as we may use some arbitration in MueLu,
    // for instance with CoupledAggregation. What we really need to do is to use the same order of processors containing
    // the same order of GIDs. To achieve that, the newly created subcommunicator must be conforming with the order. For
    // instance, if we have [{0,1,2}, {3,4,5}], we create a subcommunicator where processor 0 gets rank 0, and processor 1
    // gets rank 1. If, on the other hand, we have [{3,4,5}, {0,1,2}], we assign rank 1 to processor 0, and rank 0 to processor 1.
    // This rank permutation requires help from Epetra/Tpetra, both of which have no such API in place.
    // One should also be concerned that if we had such API in place, rank 0 in subcommunicator may no longer be rank 0 in
    // MPI_COMM_WORLD, which may lead to issues for logging.
    std::string remapAlgo = pL.get<std::string>("repartition: remap algorithm");
    const bool remapPartitions = (remapAlgo != "none");
    if (remapPartitions) {
      SubFactoryMonitor m1(*this, "DeterminePartitionPlacement", currentLevel);

      int levelID = currentLevel.GetLevelID();
      if (remapAlgo == "muelu" || remapAlgo == "muelu-hops") {
        DeterminePartitionPlacement(levelID, *A, *decomposition, numPartitions);

      } else if (remapAlgo == "zoltan2_task_map") {
        RCP<MultiVector> coords = Get< RCP<MultiVector> >(currentLevel, "Coordinates");

        DeterminePartitionPlacement1(levelID, *A, *coords, *decomposition, numPartitions);
      }
    }

    // ======================================================================================================
    // Construct importer
    // ======================================================================================================
    // At this point, the following is true:
    //  * Each processors owns 0 or 1 partitions
    //  * If a processor owns a partition, that partition number is equal to the processor rank
    //  * The decomposition vector contains the partitions ids that the corresponding GID belongs to

    ArrayRCP<const GO> decompEntries;
    if (decomposition->getLocalLength() > 0)
      decompEntries = decomposition->getData(0);

#ifdef HAVE_MUELU_DEBUG
    // Test range of partition ids
    int incorrectRank = -1;
    for (int i = 0; i < decompEntries.size(); i++)
      if (decompEntries[i] >= numProcs || decompEntries[i] < 0) {
        incorrectRank = myRank;
        break;
      }

    int incorrectGlobalRank = -1;
    maxAll(comm, incorrectRank, incorrectGlobalRank);
    TEUCHOS_TEST_FOR_EXCEPTION(incorrectGlobalRank >- 1, Exceptions::RuntimeError, "pid " + toString(incorrectGlobalRank) + " encountered a partition number is that out-of-range");
#endif

    Array<GO> myGIDs;
    myGIDs.reserve(decomposition->getLocalLength());

    // Step 0: Construct mapping
    //    part number -> GIDs I own which belong to this part
    // NOTE: my own part GIDs are not part of the map
    typedef std::map<GO, Array<GO> > map_type;
    map_type sendMap;
    for (LO i = 0; i < decompEntries.size(); i++) {
      GO id  = decompEntries[i];
      GO GID = rowMap->getGlobalElement(i);

      if (id == myRank)
        myGIDs     .push_back(GID);
      else
        sendMap[id].push_back(GID);
    }
    decompEntries = Teuchos::null;

    if (IsPrint(Statistics2)) {
      size_t numLocalKept = myGIDs.size(), numGlobalKept, numGlobalRows = A->getGlobalNumRows();
      sumAll(comm, numLocalKept, numGlobalKept);
      GetOStream(Statistics2) << "Unmoved rows: " << numGlobalKept << " / " << numGlobalRows << " (" << 100*Teuchos::as<double>(numGlobalKept)/numGlobalRows << "%)" << std::endl;
    }

    int numSend = sendMap.size(), numRecv;

    // Arrayify map keys
    Array<GO> myParts(numSend), myPart(1);
    int cnt = 0;
    myPart[0] = myRank;
    for (typename map_type::const_iterator it = sendMap.begin(); it != sendMap.end(); it++)
      myParts[cnt++] = it->first;

    // Step 1: Find out how many processors send me data
    // partsIndexBase starts from zero, as the processors ids start from zero
    GO partsIndexBase = 0;
    RCP<Map>    partsIHave  = MapFactory   ::Build(lib, Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(), myParts(), partsIndexBase, comm);
    RCP<Map>    partsIOwn   = MapFactory   ::Build(lib,                                                 numProcs,  myPart(), partsIndexBase, comm);
    RCP<Export> partsExport = ExportFactory::Build(partsIHave, partsIOwn);

    RCP<GOVector> partsISend    = Xpetra::VectorFactory<GO, LO, GO, NO>::Build(partsIHave);
    RCP<GOVector> numPartsIRecv = Xpetra::VectorFactory<GO, LO, GO, NO>::Build(partsIOwn);
    if (numSend) {
      ArrayRCP<GO> partsISendData = partsISend->getDataNonConst(0);
      for (int i = 0; i < numSend; i++)
        partsISendData[i] = 1;
    }
    (numPartsIRecv->getDataNonConst(0))[0] = 0;

    numPartsIRecv->doExport(*partsISend, *partsExport, Xpetra::ADD);
    numRecv = (numPartsIRecv->getData(0))[0];

    // Step 2: Get my GIDs from everybody else
    MPI_Datatype MpiType = MpiTypeTraits<GO>::getType();
    int msgTag = 12345;  // TODO: use Comm::dup for all internal messaging

    // Post sends
    Array<MPI_Request> sendReqs(numSend);
    cnt = 0;
    for (typename map_type::iterator it = sendMap.begin(); it != sendMap.end(); it++)
      MPI_Isend(static_cast<void*>(it->second.getRawPtr()), it->second.size(), MpiType, Teuchos::as<GO>(it->first), msgTag, *rawMpiComm, &sendReqs[cnt++]);

    // Do waits
    map_type recvMap;
    size_t totalGIDs = myGIDs.size();
    for (int i = 0; i < numRecv; i++) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, msgTag, *rawMpiComm, &status);

      // Get rank and number of elements from status
      int fromRank = status.MPI_SOURCE, count;
      MPI_Get_count(&status, MpiType, &count);

      recvMap[fromRank].resize(count);
      MPI_Recv(static_cast<void*>(recvMap[fromRank].getRawPtr()), count, MpiType, fromRank, msgTag, *rawMpiComm, &status);

      totalGIDs += count;
    }

    // Merge GIDs
    myGIDs.reserve(totalGIDs);
    for (typename map_type::const_iterator it = recvMap.begin(); it != recvMap.end(); it++) {
      int offset = myGIDs.size(), len = it->second.size();
      if (len) {
        myGIDs.resize(offset + len);
        memcpy(myGIDs.getRawPtr() + offset, it->second.getRawPtr(), len*sizeof(GO));
      }
    }
    // NOTE 2: The general sorting algorithm could be sped up by using the knowledge that original myGIDs and all received chunks
    // (i.e. it->second) are sorted. Therefore, a merge sort would work well in this situation.
    std::sort(myGIDs.begin(), myGIDs.end());

    // Step 3: Construct importer
    RCP<Map>          newRowMap      = MapFactory   ::Build(lib, rowMap->getGlobalNumElements(), myGIDs(), indexBase, origComm);
    RCP<const Import> rowMapImporter = ImportFactory::Build(rowMap, newRowMap);

    Set(currentLevel, "Importer", rowMapImporter);

    // ======================================================================================================
    // Print some data
    // ======================================================================================================
    if (pL.get<bool>("repartition: print partition distribution") && IsPrint(Statistics2)) {
      // Print the grid of processors
      GetOStream(Statistics2) << "Partition distribution over cores (ownership is indicated by '+')" << std::endl;

      char amActive = (myGIDs.size() ? 1 : 0);
      std::vector<char> areActive(numProcs, 0);
      MPI_Gather(&amActive, 1, MPI_CHAR, &areActive[0], 1, MPI_CHAR, 0, *rawMpiComm);

      int rowWidth = std::min(Teuchos::as<int>(ceil(sqrt(numProcs))), 100);
      for (int proc = 0; proc < numProcs; proc += rowWidth) {
        for (int j = 0; j < rowWidth; j++)
          if (proc + j < numProcs)
            GetOStream(Statistics2) << (areActive[proc + j] ? "+" : ".");
          else
          GetOStream(Statistics2) << " ";

        GetOStream(Statistics2) << "      " << proc << ":" << std::min(proc + rowWidth, numProcs) - 1 << std::endl;
      }
    }

  } // Build

  //----------------------------------------------------------------------
  template<typename T, typename W>
  struct Triplet {
    T    i, j;
    W    v;
  };
  template<typename T, typename W>
  static bool compareTriplets(const Triplet<T,W>& a, const Triplet<T,W>& b) {
    return (a.v > b.v); // descending order
  }

#ifdef HAVE_MUELU_RCA
  int procDist1D(int src, int dst, int dim) {
    int dist_pos, dist_neg;
    if (dst > src) {
       dist_pos = dst - src;
       dist_neg = dim - (dst - src);
    } else {
       dist_pos = dim - (src - dst);
       dist_neg = src - dst;
    }

    return std::min(dist_pos, dist_neg);
  }

  int procDistance(int proc1, int proc2) {
    mesh_coord_t coord1, coord2;

    rca_get_meshcoord(as<uint16_t>(proc1), &coord1);
    rca_get_meshcoord(as<uint16_t>(proc2), &coord2);

    return
      procDist1D(coord1.mesh_x, coord2.mesh_x, machineDims[0]) +
      procDist1D(coord1.mesh_y, coord2.mesh_y, machineDims[1]) +
      procDist1D(coord1.mesh_z, coord2.mesh_z, machineDims[2]);
  }
#endif

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  DeterminePartitionPlacement(int levelID, const Matrix& A, GOVector& decomposition, GO numPartitions) const {
    RCP<const Map> rowMap = A.getRowMap();

    RCP<const Teuchos::Comm<int> > comm = rowMap->getComm()->duplicate();
    int numProcs = comm->getSize();

    RCP<const Teuchos::MpiComm<int> > tmpic = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
    TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, Exceptions::RuntimeError, "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
    RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();

    const Teuchos::ParameterList& pL = GetParameterList();

    // maxLocal is a constant which determins the number of largest edges which are being exchanged
    // The idea is that we do not want to construct the full bipartite graph, but simply a subset of
    // it, which requires less communication. By selecting largest local edges we hope to achieve
    // similar results but at a lower cost.
    const int maxLocal = pL.get<int>("repartition: remap num values");
    const int dataSize = 2*maxLocal;

    ArrayRCP<GO> decompEntries;
    if (decomposition.getLocalLength() > 0)
      decompEntries = decomposition.getDataNonConst(0);

    // Step 1: Sort local edges by weight
    // Each edge of a bipartite graph corresponds to a triplet (i, j, v) where
    //   i: processor id that has some piece of part with part_id = j
    //   j: part id
    //   v: weight of the edge
    // We set edge weights to be the total number of nonzeros in rows on this processor which
    // correspond to this part_id. The idea is that when we redistribute matrix, this weight
    // is a good approximation of the amount of data to move.
    // We use two maps, original which maps a partition id of an edge to the corresponding weight,
    // and a reverse one, which is necessary to sort by edges.
    std::map<GO,GO> lEdges;
    for (LO i = 0; i < decompEntries.size(); i++)
      lEdges[decompEntries[i]] += A.getNumEntriesInLocalRow(i);

    std::string remapAlgo = pL.get<std::string>("repartition: remap algorithm");
    if (remapAlgo == "muelu-hops") {
#ifdef HAVE_MUELU_RCA
      int myRank   = comm->getRank();
      // Get Cray node info
      rs_node_t procInfo;
      rca_get_nodeid(&procInfo);
      int procId = as<int>(procInfo.rs_node_s._node_id);  // original field is uint32_t

      // Assemble processor ids
      std::vector<int> procIds(numProcs);
      MPI_Allgather(&procId, 1, MPI_INT, &procIds[0], 1, MPI_INT, *rawMpiComm);

      // Adjust edge weight by the distance
      // The idea is that we don't want to send data a long distance.
      for (typename std::map<GO,GO>::iterator it = lEdges.begin(); it != lEdges.end(); it++) {
        int hopDistance = procDistance(procIds[myRank], procIds[it->first]);

        it->second /= hopDistance + 1;
        // it->second /= pow(2.0, hopDistance);
      }
#else
      GetOStream(Warnings0,0) << "Using remapping \"muelu\" instead of \"muelu-hops\", as RCA is not available" << std::endl;
#endif
    }

    // Reverse map, so that edges are sorted by weight.
    // This results in multimap, as we may have edges with the same weight
    std::multimap<GO,GO> revlEdges;
    for (typename std::map<GO,GO>::const_iterator it = lEdges.begin(); it != lEdges.end(); it++)
      revlEdges.insert(std::make_pair(it->second, it->first));

    // Both lData and gData are arrays of data which we communicate. The data is stored
    // in pairs, so that data[2*i+0] is the part index, and data[2*i+1] is the corresponding edge weight.
    // We do not store processor id in data, as we can compute that by looking on the offset in the gData.
    Array<GO> lData(dataSize, -1), gData(numProcs * dataSize);
    int numEdges = 0;
    for (typename std::multimap<GO,GO>::reverse_iterator rit = revlEdges.rbegin(); rit != revlEdges.rend() && numEdges < maxLocal; rit++) {
      lData[2*numEdges+0] = rit->second; // part id
      lData[2*numEdges+1] = rit->first;  // edge weight
      numEdges++;
    }

    // Step 2: Gather most edges
    // Each processors contributes maxLocal edges by providing maxLocal pairs <part id, weight>, which is of size dataSize
    MPI_Datatype MpiType = MpiTypeTraits<GO>::getType();
    MPI_Allgather(static_cast<void*>(lData.getRawPtr()), dataSize, MpiType, static_cast<void*>(gData.getRawPtr()), dataSize, MpiType, *rawMpiComm);

    // Step 3: Construct mapping

    // Construct the set of triplets
    std::vector<Triplet<int,int> > gEdges(numProcs * maxLocal);
    size_t k = 0;
    for (LO i = 0; i < gData.size(); i += 2) {
      GO part   = gData[i+0];
      GO weight = gData[i+1];
      if (part != -1) {                     // skip nonexistent edges
        gEdges[k].i = i/dataSize;           // determine the processor by its offset (since every processor sends the same amount)
        gEdges[k].j = part;
        gEdges[k].v = weight;
        k++;
      }
    }
    gEdges.resize(k);

    // Sort edges by weight
    // NOTE: compareTriplets is actually a reverse sort, so the edges weight is in decreasing order
    std::sort(gEdges.begin(), gEdges.end(), compareTriplets<int,int>);

    // Do matching
    std::map<int,int> match;
    std::vector<char> matchedRanks(numProcs, 0), matchedParts(numProcs, 0);
    int numMatched = 0;
    for (typename std::vector<Triplet<int,int> >::const_iterator it = gEdges.begin(); it != gEdges.end(); it++) {
      GO rank = it->i;
      GO part = it->j;
      if (matchedRanks[rank] == 0 && matchedParts[part] == 0) {
        matchedRanks[rank] = 1;
        matchedParts[part] = 1;
        match[part] = rank;
        numMatched++;
      }
    }
    GetOStream(Statistics0) << "Number of unassigned paritions before cleanup stage: " << (numPartitions - numMatched) << " / " << numPartitions << std::endl;

    // Step 4: Assign unassigned partitions
    // We do that through random matching for remaining partitions. Not all part numbers are valid, but valid parts are a subset of [0, numProcs).
    // The reason it is done this way is that we don't need any extra communication, as we don't need to know which parts are valid.
    for (int part = 0, matcher = 0; part < numProcs; part++)
      if (match.count(part) == 0) {
        // Find first non-matched rank
        while (matchedRanks[matcher])
          matcher++;

        match[part] = matcher++;
      }

    // Step 5: Permute entries in the decomposition vector
    for (LO i = 0; i < decompEntries.size(); i++)
      decompEntries[i] = match[decompEntries[i]];
  }

  //----------------------------------------------------------------------

  // Returns a list of communication graph neighbors
  // NOTE: the list has self-edges first
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  getNeighborData(const Matrix& A, std::vector<int>& neighs, std::vector<SC>& weights) const {
    SC zero = Teuchos::ScalarTraits<SC>::zero(), one = Teuchos::ScalarTraits<SC>::one();

    int myRank = A.getRowMap()->getComm()->getRank();
    neighs .resize(1, myRank);
    weights.resize(1, zero);

    RCP<const Import> importer = A.getCrsGraph()->getImporter();
    if (importer != null) {
      ArrayView<const int> exportPIDs = importer->getExportPIDs();

      if (exportPIDs.size()) {
        // NOTE: exportPIDs is sorted but not unique ( 1 1 1 2 2 3 4 4 4 )
        neighs .push_back(exportPIDs[0]);
        weights.push_back(one);
        for (int i = 1; i < exportPIDs.size(); i++) {
          if (exportPIDs[i] != exportPIDs[i-1]) {
            neighs .push_back(exportPIDs[i]);
            weights.push_back(one);

          } else {
            weights.back() += one;
          }
        }
      }
    }
  }

  template<class Scalar>
  std::string coord2str(const std::vector<std::vector<Scalar> >& coords, int i) {
    std::string str = toString(coords[0][i]);
    for (size_t k = 1; k < coords.size(); k++)
      str += "," + toString(coords[k][i]);

    return str;
  }

  // Output Gnuplot data
  template<class Scalar>
  void plotCommGraph(const std::string& filename, const std::vector<std::vector<Scalar> >& coords, const std::vector<int>& ia, const std::vector<int>& ja) {
    std::string coordsFilename = filename + ".coord";
    std::string graphFilename  = filename + ".graph";

    size_t nDim = coords.size();

    // Output coordinates
    std::ofstream ofs(coordsFilename.c_str());
    ofs << std::fixed << std::setprecision(15);
    size_t numCoords = coords[0].size();
    for (size_t i = 0; i < numCoords; i++) {
      for (size_t j = 0; j < nDim; j++)
        ofs << " " << coords[j][i];
      ofs << "\n";
    }
    ofs.close();

    // Output application communication graph
    ofs.open(graphFilename.c_str());
    ofs << "splot \"" + coordsFilename + "\"\n";
    ofs << "set style arrow 5 nohead size screen 0.03,15,135 ls 1\n";
    for (size_t i = 0; i+1 < ia.size(); i++)
      for (int j = ia[i]; j < ia[i+1]; j++) {
        // We don't want to plot an arrow if it has zero length
        // NOTE: we don't do the check (i == ja[j]) as it is done
        // automatically when comparing nodes
        bool sameNode = true;
        for (size_t k = 0; k < nDim; k++)
          if (coords[k][i] != coords[k][ja[j]]) {
            sameNode = false;
            break;
          }

        if (!sameNode)
          ofs << "set arrow from " << coord2str(coords, i)     <<
                            " to " << coord2str(coords, ja[j]) << " as 5" << std::endl;
      }
    ofs << "replot\n";
    ofs << "pause -1\n";
    ofs.close();
  }

  template<class Scalar>
  void printCommGraph(Teuchos::FancyOStream& out, const std::vector<int>& ia, const std::vector<int>& ja, const std::vector<Scalar>& weights) {
    out << "BEGIN App Graph:" << std::endl;
    for (size_t i = 0; i+1 < ia.size(); i++) {
      out << "APP " << i;

      for (int j = ia[i]; j < ia[i+1]; j++)
        if (ja[j] != (int)i)
          out << " " << ja[j] << "(" << weights[j] << ")";

      out << std::endl;
    }
    out << "END   App Graph." << std::endl;
  }

  static void printMapping(Teuchos::FancyOStream& out, const std::vector<int>& procIds, const std::vector<int>& ia, const std::vector<int>& ja) {
    out << "BEGIN App to NID Mapping Info:" << std::endl;

    typedef Teuchos::TabularOutputter TTO;
    TTO outputter(out);
    outputter.pushFieldSpec("",           TTO::STRING, TTO::LEFT,  TTO::GENERAL, 3);
    outputter.pushFieldSpec("ORIG_BLOCK", TTO::INT,    TTO::RIGHT, TTO::GENERAL, 12);
    outputter.pushFieldSpec("NEW_BLOCK",  TTO::INT,    TTO::RIGHT, TTO::GENERAL, 11);
    outputter.pushFieldSpec("",           TTO::STRING, TTO::LEFT,  TTO::GENERAL, 3);
    outputter.pushFieldSpec("NODE_ID",    TTO::INT,    TTO::RIGHT, TTO::GENERAL, 9);

    outputter.outputHeader();
    for (size_t i = 0; i+1 < ia.size(); i++) {
      outputter.outputField("MAP");
      outputter.outputField(i);
      outputter.outputField(ja[ia[i]]);
      outputter.outputField("nid");
      outputter.outputField(procIds[i]);
      outputter.nextRow();
    }

    out << "END   App to NID Mapping Info." << std::endl;
  }


  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void RepartitionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::
  DeterminePartitionPlacement1(int levelID, const Matrix& A, const MultiVector& coordinates, GOVector& decomposition, GO numParts) const {
#ifndef HAVE_MUELU_ZOLTAN2
    return;

#else
    RCP<const Map> rowMap = A.getRowMap();

    RCP<const Teuchos::Comm<int> > comm  = rowMap->getComm();

    int numProcs = comm->getSize();
    int myRank   = comm->getRank();

    // Do not do anything for a single processor
    if (numProcs == 1) {
      GetOStream(Runtime0, 0) << "Skipping task remapping: single processor" << std::endl;
      return;
    }

    // Get raw communicator
    RCP<const Teuchos::MpiComm<int> > tmpic = rcp_dynamic_cast<const Teuchos::MpiComm<int> >(comm);
    TEUCHOS_TEST_FOR_EXCEPTION(tmpic == Teuchos::null, Exceptions::RuntimeError,
                               "Cannot cast base Teuchos::Comm to Teuchos::MpiComm object.");
    RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > rawMpiComm = tmpic->getRawMpiComm();


    SC zero = Teuchos::ScalarTraits<SC>::zero();

    MPI_Datatype MPI_SCType = MpiTypeTraits<SC>::getType();
    MPI_Datatype MPI_LOType = MpiTypeTraits<LO>::getType();

    size_t procDim  = 3;
    size_t coordDim = coordinates.getNumVectors();
    LO     nLocal   = as<LO>(coordinates.getLocalLength());
    TEUCHOS_TEST_FOR_EXCEPTION(nLocal == 0, Exceptions::RuntimeError,
                               "Each subdomain must contain at least one point");

    ArrayRCP<GO> decompEntries;
    if (decomposition.getLocalLength() > 0)
      decompEntries = decomposition.getDataNonConst(0);
    LO nBLocal = as<LO>(decompEntries.size());

    LO blkSize = A.GetFixedBlockSize();
    TEUCHOS_TEST_FOR_EXCEPTION(nBLocal != nLocal*blkSize, Exceptions::RuntimeError,
                               "Number of local entries in decomposition (" << nBLocal << ") is different "
                               "from the number of local entries in coordinates (" << nLocal << ") times block size (" << blkSize << ")");
    for (LO i = 0; i < nBLocal; i++)
      TEUCHOS_TEST_FOR_EXCEPTION(decompEntries[i] < 0 || decompEntries[i] >= numParts, Exceptions::RuntimeError,
                                 "Our major assumption for this remapping is that part ids are from 0 to the number of parts. "
                                 "However, we encountered part id " << decompEntries[i]);

    // Step 0: Initalize C-style arrays
    SC                           *partCoords [procDim], *procCoords [procDim];
    std::vector<std::vector<SC> > partCoordsV(procDim),  procCoordsV(procDim);
    for (size_t j = 0; j < procDim; j++) {
      partCoordsV[j].resize(numParts, zero);
      partCoords[j] = &partCoordsV[j][0];

      procCoordsV[j].resize(numProcs, zero);
      procCoords[j] = &procCoordsV[j][0];
    }

    // Step 1: Construct part communication graph from the matrix
    std::vector<int> partCommXAdj(numProcs+1), partCommAdj;
    std::vector<SC>  edgeWeights;

    // We are unable to construct the application graph in most cases. The
    // reason is that we have run a repartitioning algorithm prior to the parts
    // no longer correspond to original parts. Therefore, matrix Import is no
    // longer correct for the new partitioning. To construct an application
    // graph, we would have to either construct a permuted matrix and get its
    // Import, or to go through all local rows, and check every nonzero entry
    // about which subdomain it belongs to. That is very expensive, as we would
    // have to actually assemble new subdomains first.
    // The only case where we can easily construct the graph is when the numbers
    // of parts and processors are the same, and each processors owns its own part.
    bool haveAppGraph = true;

    if (numParts != numProcs) {
      haveAppGraph = false;

    } else {
      int haveAppGraphInt = 1;
      for (LO i = 0; i < nBLocal; i++)
        if (decompEntries[i] != myRank) {
          haveAppGraphInt = 0;
          break;
        }
      MPI_Allreduce(MPI_IN_PLACE, &haveAppGraphInt, 1, MPI_INT, MPI_MIN, *rawMpiComm);
      haveAppGraph = (haveAppGraphInt != 0);
    }

    if (haveAppGraph && blkSize > 1) {
      GetOStream(Warnings0, 0) << "Ignoring communication graph, as matrix is a block matrix" << std::endl;
      haveAppGraph = false;
    }

    if (haveAppGraph) {
      // NOTE: in the constructed graph, self edges go first
      std::vector<int> neighs;
      std::vector<SC>  weights;
      getNeighborData(A, neighs, weights);
      int numNeigh = as<int>(neighs.size());

      std::vector<int> numNeighAll(numProcs);
      MPI_Allgather(&numNeigh, 1, MPI_INT, &numNeighAll[0], 1, MPI_INT, *rawMpiComm);

      // This is not the correct version as it is shifted right by one, but this
      // exteded variant also serves as a displacement array for MPI_Allgatherv
      // call. This means that if numNeighAll = { 1 2 3 4 } then partCommXAdj =
      // { 0 1 3 6 10 } instead of { 1 3 6 10 }.
      partCommXAdj[0] = 0;
      for (int i = 0; i < numProcs; i++)
        partCommXAdj[i+1] = numNeighAll[i] + partCommXAdj[i];

      partCommAdj.resize(partCommXAdj.back());
      edgeWeights.resize(partCommXAdj.back());
      MPI_Allgatherv(&neighs [0], numNeigh, MPI_INT,    &partCommAdj[0], &numNeighAll[0], &partCommXAdj[0], MPI_INT,    *rawMpiComm);
      MPI_Allgatherv(&weights[0], numNeigh, MPI_SCType, &edgeWeights[0], &numNeighAll[0], &partCommXAdj[0], MPI_SCType, *rawMpiComm);

      printCommGraph(GetOStream(Statistics1,0), partCommXAdj, partCommAdj, edgeWeights);
    }

    // Step 2: Construct part coordinates (by averaging points in subdomains)
    // NOTE: if coordDim < procDim, we automatically pad with 0
    std::vector<LO> numPartCoords(numParts);
    for (LO i = 0; i < nLocal; i++)
      numPartCoords[decompEntries[i*blkSize]]++;

    MPI_Allreduce(MPI_IN_PLACE, &numPartCoords[0], numParts, MPI_LOType, MPI_SUM, *rawMpiComm);

    for (size_t j = 0; j < coordDim; j++) {
      ArrayRCP<const SC> coords = coordinates.getData(j);

      // NOTE: this is where we assume that part ids must be from 0 to numParts
      for (LO i = 0; i < nLocal; i++)
        partCoords[j][decompEntries[i*blkSize]] += coords[i];

      MPI_Allreduce(MPI_IN_PLACE, partCoords[j], numParts, MPI_SCType, MPI_SUM, *rawMpiComm);

      for (int i = 0; i < numParts; i++)
        partCoords[j][i] /= numPartCoords[i];
    }

    if (haveAppGraph && !myRank)
      plotCommGraph("geom_comm", partCoordsV, partCommXAdj, partCommAdj);

    // Step 3: Construct node coordinates (by calling rcs routines)
#ifdef HAVE_MUELU_RCA
    // Get Cray node info
    rs_node_t procInfo;
    rca_get_nodeid(&procInfo);
    int procId = as<int>(procInfo.rs_node_s._node_id);  // original field is uint32_t

    // Assemble processor ids
    std::vector<int> procIds(numProcs);
    MPI_Allgather(&procId, 1, MPI_INT, &procIds[0], 1, MPI_INT, *rawMpiComm);

    // Construct processor coordinates
    for (int i = 0; i < numProcs; i++) {
      mesh_coord_t procCoord;
      rca_get_meshcoord(as<uint16_t>(procIds[i]), &procCoord);

      procCoords[0][i] = as<SC>(procCoord.mesh_x);
      procCoords[1][i] = as<SC>(procCoord.mesh_y);
      procCoords[2][i] = as<SC>(procCoord.mesh_z);
    }
#else
    std::vector<int> procIds(numProcs);

    std::ifstream ifs("node_alloc.txt");
    if (ifs.is_open()) {
      assert(numProcs <= 113);
      for (int i = 0; i < numProcs; i++) {
        ifs >> procIds[i];
        ifs >> procCoords[0][i] >> procCoords[1][i] >> procCoords[2][i];
      }

    } else {
      for (int i = 0; i < numProcs; i++) {
        int procId = i;

        procCoords[0][i] = as<SC>(procId);
        procIds[i]       = procId;
      }
    }
#endif

    // Step 4: Call mapping procedure
    std::vector<int> proc2partXAdj(numProcs+1);
    std::vector<int> proc2partAdj (numParts);
    int  partArraySize = -1;
    int *partNoArray   = NULL;
    {
      SubFactoryMonitor m1(*this, "Zoltan2TaskMapper", levelID);
      if (haveAppGraph) {
        // Originally, we set partArraySize to -1. However, that uncovered a bug in Zoltan2 which
        // assigned two parts to one processor even when the number of processors and parts were the
        // same. The current expression will run MJ as RCB (Siva).
        // Recently, Mehmet committed a patch (7f271bd) which may have fixed the issue.
        partArraySize = as<int>(log(double(std::max(numProcs, numParts))) / log(2.0) + 1);

        Zoltan2::coordinateTaskMapperInterface<int,SC,SC>(comm,
                                                          procDim,  numProcs, procCoords,
                                                          coordDim, numParts, partCoords,
                                                          &partCommXAdj[1],  &partCommAdj[0],  &edgeWeights[0], // [1] is because of the right shift of array
                                                          &proc2partXAdj[1], &proc2partAdj[0],                  // see comments above
                                                          partArraySize, partNoArray, machineDims);
      } else {
        Zoltan2::coordinateTaskMapperInterface<int,SC,SC>(comm,
                                                          procDim,  numProcs, procCoords,
                                                          coordDim, numParts, partCoords,
                                                          NULL,              NULL,             NULL,
                                                          &proc2partXAdj[1], &proc2partAdj[0],                  // see comments above
                                                          partArraySize, partNoArray, machineDims);
      }
    }

    // Construct reverse mapping
    std::vector<int> part2proc(numParts, -1);
    for (int i = 0; i < numProcs; i++) {
      TEUCHOS_TEST_FOR_EXCEPTION(proc2partXAdj[i+1] - proc2partXAdj[i] > 1, Exceptions::RuntimeError, "Multiple parts per processor");

      for (int j = proc2partXAdj[i]; j < proc2partXAdj[i+1]; j++)
        part2proc[proc2partAdj[j]] = i;
    }

    // At this point the following is true:
    //   - Processor i gets parts proc2partAdj[proc2partXAdj[i]], ..., proc2partAdj[proc2partXAdj[i+1]-1]
    //   - Part      j is now owned by part2proc[j]

    if (haveAppGraph) {
      // Here we assume that original part i belonged to processor i, which is only the case after repartitioning if we first
      // try to assign parts to processors [0, ..., numParts-1]

      // In order to simplify plotting of the new communication graph, we simply need to permute the vertex coordinates
      // without permuting the graph. Originally, part i belonged to processor i and had proc coordinates procCoords[0-2][i].
      // Now it belongs to processor part2proc[i], and has proc coordinates procCoords[0-2][part2proc[i]].
      std::vector<std::vector<SC> > procCoordsNew(procDim);
      for (size_t j = 0; j < procDim; j++) {
        procCoordsNew[j].resize(numProcs);
        for (int i = 0; i < numProcs; i++)
          procCoordsNew[j][i] = procCoordsV[j][part2proc[i]];
      }

      if (!myRank) {
        plotCommGraph("proc_comm_new", procCoordsNew, partCommXAdj, partCommAdj);
        printMapping(GetOStream(Statistics1,0), procIds, proc2partXAdj, proc2partAdj);
      }
    }

    // Step 5: Permute entries in the decomposition vector
    // Decomposition vector means that entry i belongs to part decompEntries[i] which goes to processor decompEntries[i].
    // Part j now actually belongs to processor part2proc[j], so it should go there.
    for (LO i = 0; i < decompEntries.size(); i++)
      decompEntries[i] = part2proc[decompEntries[i]];
#endif
  }

} // namespace MueLu

#endif //ifdef HAVE_MPI

#endif // MUELU_REPARTITIONFACTORY_DEF_HPP
