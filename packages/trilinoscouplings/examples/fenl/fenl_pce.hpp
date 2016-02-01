/*
// ************************************************************************
//
//   Kokkos: Manycore Performance-Portable Multidimensional Arrays
//              Copyright (2012) Sandia Corporation
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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
*/

#include "Stokhos_Tpetra_UQ_PCE.hpp"
#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"

#if defined( HAVE_STOKHOS_BELOS )
#include "Belos_TpetraAdapter_UQ_PCE.hpp"
#endif

#if defined( HAVE_STOKHOS_MUELU )
#include "Stokhos_MueLu_UQ_PCE.hpp"
#endif

#include <Kokkos_Core.hpp>
#include <HexElement.hpp>
#include <fenl.hpp>
#include <fenl_functors_pce.hpp>
#include <fenl_impl.hpp>

namespace Kokkos {
namespace Example {

#if defined( KOKKOS_USING_EXPERIMENTAL_VIEW )

  //! Get mean values matrix for mean-based preconditioning
  /*! Specialization for Sacado::UQ::PCE
   */
  template <class Storage, class ... P>
  class GetMeanValsFunc< Kokkos::View< Sacado::UQ::PCE<Storage>*,
                                       P... > > {
  public:
    typedef Sacado::UQ::PCE<Storage> Scalar;
    typedef Kokkos::View< Scalar*, P... > ViewType;
    typedef ViewType MeanViewType;
    typedef typename ViewType::execution_space execution_space;
    typedef typename ViewType::size_type size_type;

    GetMeanValsFunc(const ViewType& vals_) : vals(vals_)
    {
      const size_type nnz = vals.dimension_0();
      mean_vals =
        Kokkos::make_view<ViewType>("mean-values", Kokkos::cijk(vals), nnz, 1);
      Kokkos::parallel_for( nnz, *this );
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const size_type i) const
    {
      mean_vals(i) = vals(i).fastAccessCoeff(0);
    }

     MeanViewType getMeanValues() const { return mean_vals; }

  private:
    MeanViewType mean_vals;
    ViewType vals;
  };

#else

  //! Get mean values matrix for mean-based preconditioning
  /*! Specialization for Sacado::UQ::PCE
   */
  template <class Storage, class Layout, class Memory, class Device>
  class GetMeanValsFunc< Kokkos::View< Sacado::UQ::PCE<Storage>*,
                                       Layout, Memory, Device > > {
  public:
    typedef Sacado::UQ::PCE<Storage> Scalar;
    typedef Kokkos::View< Scalar*, Layout, Memory, Device > ViewType;
    typedef ViewType MeanViewType;
    typedef typename ViewType::execution_space execution_space;
    typedef typename ViewType::size_type size_type;

    GetMeanValsFunc(const ViewType& vals_) : vals(vals_)
    {
      const size_type nnz = vals.dimension_0();
      mean_vals = ViewType("mean-values", vals.cijk(), nnz, 1);
      Kokkos::parallel_for( nnz, *this );
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const size_type i) const
    {
      mean_vals(i) = vals(i).fastAccessCoeff(0);
    }

     MeanViewType getMeanValues() const { return mean_vals; }

  private:
    MeanViewType mean_vals;
    ViewType vals;
  };

#endif

}
}
