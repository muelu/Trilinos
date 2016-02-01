#ifndef _KOKKOS_GRAPH_COLOR_HPP
#define _KOKKOS_GRAPH_COLOR_HPP

#include "KokkosKernels_GraphColor_impl.hpp"
#include "KokkosKernels_GraphColorHandle.hpp"
#include "KokkosKernels_Utils.hpp"
namespace KokkosKernels{

namespace Experimental{


namespace Graph{

template <class KernelHandle,typename in_row_index_view_type, typename in_nonzero_index_view_type>
void graph_color_symbolic(
    KernelHandle *handle,
    typename KernelHandle::row_index_type num_rows,
    typename KernelHandle::row_index_type num_cols,
    in_row_index_view_type row_map,
    in_nonzero_index_view_type entries,
    bool is_symmetric = true){

  Kokkos::Impl::Timer timer;

  typename KernelHandle::GraphColoringHandleType *gch = handle->get_graph_coloring_handle();

  ColoringAlgorithm algorithm = gch->get_coloring_type();

  typedef typename KernelHandle::GraphColoringHandleType::color_view_type color_view_type;
  color_view_type colors_out = color_view_type("Graph Colors", num_rows);

  typedef typename Impl::GraphColor
      <typename KernelHandle::GraphColoringHandleType, in_row_index_view_type, in_nonzero_index_view_type> BaseGraphColoring;
  BaseGraphColoring *gc = NULL;


  switch (algorithm){
  case COLORING_SERIAL:

    gc = new BaseGraphColoring(
        num_rows, entries.dimension_0(),
        row_map, entries, gch);
    break;
  case COLORING_VB:
  case COLORING_VBBIT:
  case COLORING_VBCS:

    typedef typename Impl::GraphColor_VB
        <typename KernelHandle::GraphColoringHandleType, in_row_index_view_type, in_nonzero_index_view_type> VBGraphColoring;
    gc = new VBGraphColoring(
        num_rows, entries.dimension_0(),
        row_map, entries, gch);
    break;
  case COLORING_EB:

    typedef typename Impl::GraphColor_EB
        <typename KernelHandle::GraphColoringHandleType, in_row_index_view_type, in_nonzero_index_view_type> EBGraphColoring;

    gc = new EBGraphColoring(num_rows, entries.dimension_0(),row_map, entries, gch);
    break;
  case COLORING_DEFAULT:
    break;

  }

  int num_phases = 0;
  gc->color_graph(colors_out, num_phases);
  delete gc;
  double coloring_time = timer.seconds();
  gch->add_to_overall_coloring_time(coloring_time);
  gch->set_coloring_time(coloring_time);
  gch->set_num_phases(num_phases);
  gch->set_vertex_colors(colors_out);
}

}
}
}

#endif//_KOKKOS_GRAPH_COLOR_HPP
