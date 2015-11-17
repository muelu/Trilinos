// @HEADER
//
// ***********************************************************************
//
//             Xpetra: A linear algebra interface package
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
//                    Tobias Wiesner    (tawiesn@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

#ifndef PACKAGES_XPETRA_SUP_UTILS_XPETRA_IO_HPP_
#define PACKAGES_XPETRA_SUP_UTILS_XPETRA_IO_HPP_

#include "Xpetra_ConfigDefs.hpp"

#ifdef HAVE_XPETRA_EPETRA
# ifdef HAVE_MPI
#  include "Epetra_MpiComm.h"
# endif
#endif

#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_RowMatrixOut.h>
#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_MultiVectorIn.h>
#include <EpetraExt_BlockMapIn.h>
#include <Xpetra_EpetraUtils.hpp>
#include <Xpetra_EpetraMultiVector.hpp>
#include <EpetraExt_BlockMapOut.h>
#endif

#ifdef HAVE_XPETRA_TPETRA
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <Xpetra_TpetraMultiVector.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <Xpetra_TpetraBlockCrsMatrix.hpp>
#endif

#ifdef HAVE_XPETRA_EPETRA
#include <Xpetra_EpetraMap.hpp>
#endif

#include "Xpetra_Matrix.hpp"
#include "Xpetra_MatrixMatrix.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"

#include "Xpetra_Map.hpp"
#include "Xpetra_StridedMap.hpp"
#include "Xpetra_StridedMapFactory.hpp"
#include "Xpetra_MapExtractor.hpp"
#include "Xpetra_MatrixFactory.hpp"



namespace Xpetra {


#ifdef HAVE_XPETRA_EPETRA
//This non-member templated function exists so that the matrix-matrix multiply will compile if Epetra, Tpetra, and ML are enabled.
template<class SC,class LO,class GO,class NO>
RCP<Xpetra::CrsMatrixWrap<SC,LO,GO,NO> >
Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap (RCP<Epetra_CrsMatrix> &epAB)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, Exceptions::RuntimeError, "Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap cannot be used with Scalar != double, LocalOrdinal != int, GlobalOrdinal != int");
}

typedef KokkosClassic::DefaultNode::DefaultNodeType KDNT;

//specialization for the case of ScalarType=double and LocalOrdinal=GlobalOrdinal=int
template<>
inline RCP<Xpetra::CrsMatrixWrap<double,int,int,KDNT> > Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap<double,int,int,KDNT > (RCP<Epetra_CrsMatrix> &epAB) {
  RCP<Xpetra::EpetraCrsMatrixT<int,KDNT> > tmpC1 = rcp(new Xpetra::EpetraCrsMatrixT<int,KDNT>(epAB));
  RCP<Xpetra::CrsMatrix<double,int,int,KDNT> > tmpC2 = Teuchos::rcp_implicit_cast<Xpetra::CrsMatrix<double,int,int,KDNT> >(tmpC1);
  RCP<Xpetra::CrsMatrixWrap<double,int,int,KDNT> > tmpC3 = rcp(new Xpetra::CrsMatrixWrap<double,int,int,KDNT>(tmpC2));
  return tmpC3;
}
#endif

/*!
    @class IO
    @brief Xpetra utility class containing IO routines to read/write vectors, matrices etc...
 */
template <class Scalar,
class LocalOrdinal  = int,
class GlobalOrdinal = LocalOrdinal,
class Node          = KokkosClassic::DefaultNode::DefaultNodeType>
class IO {

private:
#undef XPETRA_IO_SHORT
#include "Xpetra_UseShortNames.hpp"

public:

#ifdef HAVE_XPETRA_EPETRA
  //! Helper utility to pull out the underlying Epetra objects from an Xpetra object
  // @{
  /*static RCP<const Epetra_MultiVector>                    MV2EpetraMV(RCP<MultiVector> const Vec);
    static RCP<      Epetra_MultiVector>                    MV2NonConstEpetraMV(RCP<MultiVector> Vec);

    static const Epetra_MultiVector&                        MV2EpetraMV(const MultiVector& Vec);
    static       Epetra_MultiVector&                        MV2NonConstEpetraMV(MultiVector& Vec);

    static RCP<const Epetra_CrsMatrix>                      Op2EpetraCrs(RCP<const Matrix> Op);
    static RCP<      Epetra_CrsMatrix>                      Op2NonConstEpetraCrs(RCP<Matrix> Op);

    static const Epetra_CrsMatrix&                          Op2EpetraCrs(const Matrix& Op);
    static       Epetra_CrsMatrix&                          Op2NonConstEpetraCrs(Matrix& Op);*/

  static const Epetra_Map&  Map2EpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map) {
    RCP<const Xpetra::EpetraMapT<GlobalOrdinal,Node> > xeMap = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >(Teuchos::rcpFromRef(map));
    if (xeMap == Teuchos::null)
      throw Exceptions::BadCast("Utils::Map2EpetraMap : Cast from Xpetra::Map to Xpetra::EpetraMap failed");
    return xeMap->getEpetra_Map();
  }
  // @}
#endif

#ifdef HAVE_XPETRA_TPETRA
  //! Helper utility to pull out the underlying Tpetra objects from an Xpetra object
  // @{
  /*static RCP<const Tpetra::MultiVector<SC,LO,GO,NO> >     MV2TpetraMV(RCP<MultiVector> const Vec);
    static RCP<      Tpetra::MultiVector<SC,LO,GO,NO> >     MV2NonConstTpetraMV(RCP<MultiVector> Vec);
    static RCP<      Tpetra::MultiVector<SC,LO,GO,NO> >     MV2NonConstTpetraMV2(MultiVector& Vec);

    static const Tpetra::MultiVector<SC,LO,GO,NO>&          MV2TpetraMV(const MultiVector& Vec);
    static       Tpetra::MultiVector<SC,LO,GO,NO>&          MV2NonConstTpetraMV(MultiVector& Vec);

    static RCP<const Tpetra::CrsMatrix<SC,LO,GO,NO> >       Op2TpetraCrs(RCP<const Matrix> Op);
    static RCP<      Tpetra::CrsMatrix<SC,LO,GO,NO> >       Op2NonConstTpetraCrs(RCP<Matrix> Op);

    static const Tpetra::CrsMatrix<SC,LO,GO,NO>&            Op2TpetraCrs(const Matrix& Op);
    static       Tpetra::CrsMatrix<SC,LO,GO,NO>&            Op2NonConstTpetraCrs(Matrix& Op);

    static RCP<const Tpetra::RowMatrix<SC,LO,GO,NO> >       Op2TpetraRow(RCP<const Matrix> Op);
    static RCP<      Tpetra::RowMatrix<SC,LO,GO,NO> >       Op2NonConstTpetraRow(RCP<Matrix> Op);*/


  static const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > Map2TpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map) {
    const RCP<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> >& tmp_TMap = Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> >(rcpFromRef(map));
    if (tmp_TMap == Teuchos::null)
      throw Exceptions::BadCast("Utils::Map2TpetraMap : Cast from Xpetra::Map to Xpetra::TpetraMap failed");
    return tmp_TMap->getTpetra_Map();
  }
#endif


  //! Read/Write methods
  //@{
  /*! @brief Save map to file. */
  static void Write(const std::string& fileName, const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> & M) {
    RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > tmp_Map = rcpFromRef(M);
#ifdef HAVE_XPETRA_EPETRAEXT
    const RCP<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >& tmp_EMap = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >(tmp_Map);
    if (tmp_EMap != Teuchos::null) {
#ifdef HAVE_XPETRA_EPETRAEXT
      int rv = EpetraExt::BlockMapToMatrixMarketFile(fileName.c_str(), tmp_EMap->getEpetra_Map());
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::BlockMapToMatrixMarketFile() return value of " + Teuchos::toString(rv));
#else
      throw(Exceptions::RuntimeError("Compiled without EpetraExt"));
#endif
      return;
    }
#endif // HAVE_XPETRA_EPETRAEXT

#ifdef HAVE_XPETRA_TPETRA
    const RCP<const Xpetra::TpetraMap<LocalOrdinal, GlobalOrdinal, Node> > &tmp_TMap =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMap<LocalOrdinal, GlobalOrdinal, Node> >(tmp_Map);
    if (tmp_TMap != Teuchos::null) {
      RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > TMap = tmp_TMap->getTpetra_Map();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeMapFile(fileName, *TMap);
      return;
    }
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraMap or TpetraMap in map writing");

  } //Write

  /*! @brief Save vector to file in Matrix Market format.  */
  static void Write(const std::string& fileName, const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> & Vec) {
    std::string mapfile = "map_" + fileName;
    Write(mapfile, *(Vec.getMap()));

    RCP<const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmp_Vec = Teuchos::rcpFromRef(Vec);
#ifdef HAVE_XPETRA_EPETRA
    const RCP<const Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> >& tmp_EVec = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> >(tmp_Vec);
    if (tmp_EVec != Teuchos::null) {
#ifdef HAVE_XPETRA_EPETRAEXT
      int rv = EpetraExt::MultiVectorToMatrixMarketFile(fileName.c_str(), *(tmp_EVec->getEpetra_MultiVector()));
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::RowMatrixToMatrixMarketFile return value of " + Teuchos::toString(rv));
#else
      throw Exceptions::RuntimeError("Compiled without EpetraExt");
#endif
      return;
    }
#endif // HAVE_XPETRA_EPETRAEXT

#ifdef HAVE_XPETRA_TPETRA
    const RCP<const Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &tmp_TVec =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmp_Vec);
    if (tmp_TVec != Teuchos::null) {
      RCP<const Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > TVec = tmp_TVec->getTpetra_MultiVector();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeDenseFile(fileName, TVec);
      return;
    }
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraMultiVector or TpetraMultiVector in multivector writing");

  } //Write



  /*! @brief Save matrix to file in Matrix Market format. */
  static void Write(const std::string& fileName, const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> & Op) {

    Write("rowmap_"    + fileName, *(Op.getRowMap()));
    Write("colmap_"    + fileName, *(Op.getColMap()));
    Write("domainmap_" + fileName, *(Op.getDomainMap()));
    Write("rangemap_"  + fileName, *(Op.getRangeMap()));

    const Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>& crsOp =
        dynamic_cast<const Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>&>(Op);
    RCP<const Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmp_CrsMtx = crsOp.getCrsMatrix();
#if defined(HAVE_XPETRA_EPETRA)
    const RCP<const Xpetra::EpetraCrsMatrixT<GlobalOrdinal,Node> >& tmp_ECrsMtx = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraCrsMatrixT<GlobalOrdinal,Node> >(tmp_CrsMtx);
    if (tmp_ECrsMtx != Teuchos::null) {
#if defined(HAVE_XPETRA_EPETRAEXT)
      RCP<const Epetra_CrsMatrix> A = tmp_ECrsMtx->getEpetra_CrsMatrix();
      int rv = EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(), *A);
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::RowMatrixToMatrixMarketFile return value of " + Teuchos::toString(rv));
#else
      throw Exceptions::RuntimeError("Compiled without EpetraExt");
#endif
      return;
    }
#endif

#ifdef HAVE_XPETRA_TPETRA
    const RCP<const Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& tmp_TCrsMtx =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmp_CrsMtx);
    if (tmp_TCrsMtx != Teuchos::null) {
      RCP<const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A = tmp_TCrsMtx->getTpetra_CrsMatrix();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeSparseFile(fileName, A);
      return;
    }
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraCrsMatrix or TpetraCrsMatrix in matrix writing");
  } //Write

  //! @brief Read matrix from file in Matrix Market or binary format.
  static Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Read(const std::string& fileName, Xpetra::UnderlyingLib lib, const RCP<const Teuchos::Comm<int> >& comm, bool binary = false) {
    if (binary == false) {
      // Matrix Market file format (ASCII)
      if (lib == Xpetra::UseEpetra) {
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
        Epetra_CrsMatrix *eA;
        const RCP<const Epetra_Comm> epcomm = Xpetra::toEpetra(comm);
        int rv = EpetraExt::MatrixMarketFileToCrsMatrix(fileName.c_str(), *epcomm, eA);
        if (rv != 0)
          throw Exceptions::RuntimeError("EpetraExt::MatrixMarketFileToCrsMatrix return value of " + Teuchos::toString(rv));

        RCP<Epetra_CrsMatrix> tmpA = rcp(eA);

        RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A =
            Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA);
        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
      } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
        typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;

        typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type> reader_type;

        //RCP<Node> node = Xpetra::DefaultPlatform::getDefaultPlatform().getNode();
        Teuchos::ParameterList pl = Teuchos::ParameterList();
        RCP<Node> node = rcp(new Node(pl));
        bool callFillComplete = true;

        RCP<sparse_matrix_type> tA = reader_type::readSparseFile(fileName, comm, node, callFillComplete);

        if (tA.is_null())
          throw Exceptions::RuntimeError("The Tpetra::CrsMatrix returned from readSparseFile() is null.");

        RCP<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmpA1 = rcp(new Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tA));
        RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >       tmpA2 = Teuchos::rcp_implicit_cast<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmpA1);
        RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >          A     = rcp(new Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA2));

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
      } else {
        throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
      }
    } else {
      // Custom file format (binary)
      std::ifstream ifs(fileName.c_str(), std::ios::binary);
      TEUCHOS_TEST_FOR_EXCEPTION(!ifs.good(), Exceptions::RuntimeError, "Can not read \"" << fileName << "\"");
      int m, n, nnz;
      ifs.read(reinterpret_cast<char*>(&m),   sizeof(m));
      ifs.read(reinterpret_cast<char*>(&n),   sizeof(n));
      ifs.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

      int myRank = comm->getRank();

      GO indexBase = 0;
      RCP<Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >    rowMap = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, m, (myRank == 0 ? m : 0), indexBase, comm), rangeMap  = rowMap;
      RCP<Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >    colMap = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, n, (myRank == 0 ? n : 0), indexBase, comm), domainMap = colMap;
      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A   = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(rowMap, colMap, 1);

      TEUCHOS_TEST_FOR_EXCEPTION(sizeof(int) != sizeof(GO), Exceptions::RuntimeError, "Incompatible sizes");

      if (myRank == 0) {
        Teuchos::Array<GlobalOrdinal> inds;
        Teuchos::Array<Scalar> vals;
        for (int i = 0; i < m; i++) {
          int row, rownnz;
          ifs.read(reinterpret_cast<char*>(&row),    sizeof(row));
          ifs.read(reinterpret_cast<char*>(&rownnz), sizeof(rownnz));
          inds.resize(rownnz);
          vals.resize(rownnz);
          for (int j = 0; j < rownnz; j++) {
            int index;
            ifs.read(reinterpret_cast<char*>(&index), sizeof(index));
            inds[j] = Teuchos::as<GlobalOrdinal>(index);
          }
          for (int j = 0; j < rownnz; j++) {
            double value;
            ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
            vals[j] = Teuchos::as<SC>(value);
          }
          A->insertGlobalValues(row, inds, vals);
        }
      }

      A->fillComplete(domainMap, rangeMap);

      return A;
    }


  } //Read()


  /*! @brief Read matrix from file in Matrix Market or binary format.

        If only rowMap is specified, then it is used for the domainMap and rangeMap, as well.
   */
  static Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Read(const std::string&   filename,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > rowMap,
      RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > colMap           = Teuchos::null,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > domainMap        = Teuchos::null,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > rangeMap         = Teuchos::null,
      const bool           callFillComplete = true,
      const bool           binary           = false,
      const bool           tolerant         = false,
      const bool           debug            = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(rowMap.is_null(), Exceptions::RuntimeError, "Utils::Read() : rowMap cannot be null");

    RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > domain = (domainMap.is_null() ? rowMap : domainMap);
    RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > range  = (rangeMap .is_null() ? rowMap : rangeMap);

    const Xpetra::UnderlyingLib lib = rowMap->lib();
    if (binary == false) {
      if (lib == Xpetra::UseEpetra) {
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
        Epetra_CrsMatrix *eA;
        const RCP<const Epetra_Comm> epcomm = Xpetra::toEpetra(rowMap->getComm());
        const Epetra_Map& epetraRowMap    = Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*rowMap);
        const Epetra_Map& epetraDomainMap = (domainMap.is_null() ? epetraRowMap : Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*domainMap));
        const Epetra_Map& epetraRangeMap  = (rangeMap .is_null() ? epetraRowMap : Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*rangeMap));
        int rv;
        if (colMap.is_null()) {
          rv = EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), epetraRowMap, epetraRangeMap, epetraDomainMap, eA);

        } else {
          const Epetra_Map& epetraColMap  = Map2EpetraMap(*colMap);
          rv = EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), epetraRowMap, epetraColMap, epetraRangeMap, epetraDomainMap, eA);
        }

        if (rv != 0)
          throw Exceptions::RuntimeError("EpetraExt::MatrixMarketFileToCrsMatrix return value of " + Teuchos::toString(rv));

        RCP<Epetra_CrsMatrix> tmpA = rcp(eA);
        RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >           A    = Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA);

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
      } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
        typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
        typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>             reader_type;
        typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>               map_type;

        const RCP<const map_type> tpetraRowMap    = Map2TpetraMap(*rowMap);
        RCP<const map_type>       tpetraColMap    = (colMap.is_null()    ? Teuchos::null : Map2TpetraMap(*colMap));
        const RCP<const map_type> tpetraRangeMap  = (rangeMap.is_null()  ? tpetraRowMap  : Map2TpetraMap(*rangeMap));
        const RCP<const map_type> tpetraDomainMap = (domainMap.is_null() ? tpetraRowMap  : Map2TpetraMap(*domainMap));

        RCP<sparse_matrix_type> tA = reader_type::readSparseFile(filename, tpetraRowMap, tpetraColMap, tpetraDomainMap, tpetraRangeMap,
            callFillComplete, tolerant, debug);
        if (tA.is_null())
          throw Exceptions::RuntimeError("The Tpetra::CrsMatrix returned from readSparseFile() is null.");

        RCP<Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmpA1 = rcp(new Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tA));
        RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >       tmpA2 = Teuchos::rcp_implicit_cast<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(tmpA1);
        RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >          A     = rcp(new Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tmpA2));

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
      } else {
        throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
      }
    } else {
      // Custom file format (binary)
      std::ifstream ifs(filename.c_str(), std::ios::binary);
      TEUCHOS_TEST_FOR_EXCEPTION(!ifs.good(), Exceptions::RuntimeError, "Can not read \"" << filename << "\"");
      int m, n, nnz;
      ifs.read(reinterpret_cast<char*>(&m),   sizeof(m));
      ifs.read(reinterpret_cast<char*>(&n),   sizeof(n));
      ifs.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

      RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > A = Xpetra::MatrixFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap, colMap, 1);

      TEUCHOS_TEST_FOR_EXCEPTION(sizeof(int) != sizeof(GO), Exceptions::RuntimeError, "Incompatible sizes");

      Teuchos::ArrayView<const GlobalOrdinal> rowElements = rowMap->getNodeElementList();
      Teuchos::ArrayView<const GlobalOrdinal> colElements = colMap->getNodeElementList();

      Teuchos::Array<GlobalOrdinal> inds;
      Teuchos::Array<Scalar> vals;
      for (int i = 0; i < m; i++) {
        int row, rownnz;
        ifs.read(reinterpret_cast<char*>(&row),    sizeof(row));
        ifs.read(reinterpret_cast<char*>(&rownnz), sizeof(rownnz));
        inds.resize(rownnz);
        vals.resize(rownnz);
        for (int j = 0; j < rownnz; j++) {
          int index;
          ifs.read(reinterpret_cast<char*>(&index), sizeof(index));
          inds[j] = colElements[Teuchos::as<LocalOrdinal>(index)];
        }
        for (int j = 0; j < rownnz; j++) {
          double value;
          ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
          vals[j] = Teuchos::as<SC>(value);
        }
        A->insertGlobalValues(rowElements[row], inds, vals);
      }
      A->fillComplete(domainMap, rangeMap);
      return A;
    }

  }
  //@}


  static RCP<MultiVector> ReadMultiVector (const std::string& fileName, const RCP<const Map>& map) {
    Xpetra::UnderlyingLib lib = map->lib();

    if (lib == Xpetra::UseEpetra) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, ::Xpetra::Exceptions::BadCast, "Epetra can only be used with Scalar=double and Ordinal=int");

    } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
      typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
      typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>                          reader_type;
      typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>                            map_type;
      typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>            multivector_type;

      RCP<const map_type>   temp = toTpetra(map);
      RCP<multivector_type> TMV  = reader_type::readDenseFile(fileName,map->getComm(),map->getNode(),temp);
      RCP<MultiVector>      rmv  = Xpetra::toXpetra(TMV);
      return rmv;
#else
  throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
    } else {
      throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
    }


  }


  static RCP<const Map>   ReadMap         (const std::string& fileName, Xpetra::UnderlyingLib lib, const RCP<const Teuchos::Comm<int> >& comm) {
    if (lib == Xpetra::UseEpetra) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, ::Xpetra::Exceptions::BadCast, "Epetra can only be used with Scalar=double and Ordinal=int");
    } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
      typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
      typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>                          reader_type;

      RCP<Node> node = rcp(new Node());

      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > tMap = reader_type::readMapFile(fileName, comm, node);
      if (tMap.is_null())
        throw Exceptions::RuntimeError("The Tpetra::Map returned from readSparseFile() is null.");

      return Xpetra::toXpetra(tMap);
#else
      throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
    } else {
      throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
    }


  }


};


/*!
    @class IO
    @brief Xpetra utility class containing IO routines to read/write vectors, matrices.

    Specialization for LO=GO=int

    TODO: do we need specialization for SC=double and std::complex<>???
 */
template <class Scalar,class Node>
class IO<Scalar,int,int,Node> {
public:
  typedef int LocalOrdinal;
  typedef int GlobalOrdinal;

#ifdef HAVE_XPETRA_EPETRA
  //! Helper utility to pull out the underlying Epetra objects from an Xpetra object
  // @{
  static const Epetra_Map&  Map2EpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map) {
    RCP<const Xpetra::EpetraMapT<GlobalOrdinal,Node> > xeMap = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >(Teuchos::rcpFromRef(map));
    if (xeMap == Teuchos::null)
      throw Exceptions::BadCast("IO::Map2EpetraMap : Cast from Xpetra::Map to Xpetra::EpetraMap failed");
    return xeMap->getEpetra_Map();
  }
  // @}
#endif

#ifdef HAVE_XPETRA_TPETRA
  //! Helper utility to pull out the underlying Tpetra objects from an Xpetra object
  // @{
  static const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > Map2TpetraMap(const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>& map) {
    const RCP<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> >& tmp_TMap = Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMap<LocalOrdinal,GlobalOrdinal,Node> >(rcpFromRef(map));
    if (tmp_TMap == Teuchos::null)
      throw Exceptions::BadCast("IO::Map2TpetraMap : Cast from Xpetra::Map to Xpetra::TpetraMap failed");
    return tmp_TMap->getTpetra_Map();
  }
#endif


  //! Read/Write methods
  //@{
  /*! @brief Save map to file. */
  static void Write(const std::string& fileName, const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> & M) {
    RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > tmp_Map = rcpFromRef(M);
#ifdef HAVE_XPETRA_EPETRA
    const RCP<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >& tmp_EMap = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMapT<GlobalOrdinal,Node> >(tmp_Map);
    if (tmp_EMap != Teuchos::null) {
#ifdef HAVE_XPETRA_EPETRAEXT
      int rv = EpetraExt::BlockMapToMatrixMarketFile(fileName.c_str(), tmp_EMap->getEpetra_Map());
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::BlockMapToMatrixMarketFile() return value of " + Teuchos::toString(rv));
#else
      throw(Exceptions::RuntimeError("Compiled without EpetraExt"));
#endif
      return;
    }
#endif // HAVE_XPETRA_EPETRA

#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
    const RCP<const Xpetra::TpetraMap<LocalOrdinal, GlobalOrdinal, Node> > &tmp_TMap =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMap<LocalOrdinal, GlobalOrdinal, Node> >(tmp_Map);
    if (tmp_TMap != Teuchos::null) {
      RCP<const Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > TMap = tmp_TMap->getTpetra_Map();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeMapFile(fileName, *TMap);
      return;
    }
#endif // HAVE_XPETRA_TPETRA_INST_INT_INT
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraMap or TpetraMap in map writing");

  } //Write

  /*! @brief Save vector to file in Matrix Market format.  */
  static void Write(const std::string& fileName, const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> & Vec) {
    std::string mapfile = "map_" + fileName;
    Write(mapfile, *(Vec.getMap()));

    RCP<const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmp_Vec = Teuchos::rcpFromRef(Vec);
#ifdef HAVE_XPETRA_EPETRA
    const RCP<const Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> >& tmp_EVec = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraMultiVectorT<GlobalOrdinal,Node> >(tmp_Vec);
    if (tmp_EVec != Teuchos::null) {
#ifdef HAVE_XPETRA_EPETRAEXT
      int rv = EpetraExt::MultiVectorToMatrixMarketFile(fileName.c_str(), *(tmp_EVec->getEpetra_MultiVector()));
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::RowMatrixToMatrixMarketFile return value of " + Teuchos::toString(rv));
#else
      throw Exceptions::RuntimeError("Compiled without EpetraExt");
#endif
      return;
    }
#endif // HAVE_XPETRA_EPETRAEXT

#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
    const RCP<const Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &tmp_TVec =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmp_Vec);
    if (tmp_TVec != Teuchos::null) {
      RCP<const Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> > TVec = tmp_TVec->getTpetra_MultiVector();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeDenseFile(fileName, TVec);
      return;
    }
#endif // HAVE_XPETRA_TPETRA_INST_INT_INT
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraMultiVector or TpetraMultiVector in multivector writing");

  } //Write



  /*! @brief Save matrix to file in Matrix Market format. */
  static void Write(const std::string& fileName, const Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> & Op) {

    Write("rowmap_"    + fileName, *(Op.getRowMap()));
    Write("colmap_"    + fileName, *(Op.getColMap()));
    Write("domainmap_" + fileName, *(Op.getDomainMap()));
    Write("rangemap_"  + fileName, *(Op.getRangeMap()));

    const Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>& crsOp =
        dynamic_cast<const Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>&>(Op);
    RCP<const Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmp_CrsMtx = crsOp.getCrsMatrix();
#if defined(HAVE_XPETRA_EPETRA)
    const RCP<const Xpetra::EpetraCrsMatrixT<GlobalOrdinal,Node> >& tmp_ECrsMtx = Teuchos::rcp_dynamic_cast<const Xpetra::EpetraCrsMatrixT<GlobalOrdinal,Node> >(tmp_CrsMtx);
    if (tmp_ECrsMtx != Teuchos::null) {
#if defined(HAVE_XPETRA_EPETRAEXT)
      RCP<const Epetra_CrsMatrix> A = tmp_ECrsMtx->getEpetra_CrsMatrix();
      int rv = EpetraExt::RowMatrixToMatrixMarketFile(fileName.c_str(), *A);
      if (rv != 0)
        throw Exceptions::RuntimeError("EpetraExt::RowMatrixToMatrixMarketFile return value of " + Teuchos::toString(rv));
#else
      throw Exceptions::RuntimeError("Compiled without EpetraExt");
#endif // HAVE_XPETRA_EPETRAEXT
      return;
    }
#endif // endif HAVE_XPETRA_EPETRA

#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
    const RCP<const Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >& tmp_TCrsMtx =
        Teuchos::rcp_dynamic_cast<const Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmp_CrsMtx);
    if (tmp_TCrsMtx != Teuchos::null) {
      RCP<const Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A = tmp_TCrsMtx->getTpetra_CrsMatrix();
      Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >::writeSparseFile(fileName, A);
      return;
    }
#endif // HAVE_XPETRA_TPETRA_INST_INT_INT
#endif // HAVE_XPETRA_TPETRA

    throw Exceptions::BadCast("Could not cast to EpetraCrsMatrix or TpetraCrsMatrix in matrix writing");
  } //Write

  //! @brief Read matrix from file in Matrix Market or binary format.
  static Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Read(const std::string& fileName, Xpetra::UnderlyingLib lib, const RCP<const Teuchos::Comm<int> >& comm, bool binary = false) {
    if (binary == false) {
      // Matrix Market file format (ASCII)
      if (lib == Xpetra::UseEpetra) {
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
        Epetra_CrsMatrix *eA;
        const RCP<const Epetra_Comm> epcomm = Xpetra::toEpetra(comm);
        int rv = EpetraExt::MatrixMarketFileToCrsMatrix(fileName.c_str(), *epcomm, eA);
        if (rv != 0)
          throw Exceptions::RuntimeError("EpetraExt::MatrixMarketFileToCrsMatrix return value of " + Teuchos::toString(rv));

        RCP<Epetra_CrsMatrix> tmpA = rcp(eA);

        RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A =
            Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA);
        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
      } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
        typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;

        typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type> reader_type;

        //RCP<Node> node = Xpetra::DefaultPlatform::getDefaultPlatform().getNode();
        Teuchos::ParameterList pl = Teuchos::ParameterList();
        RCP<Node> node = rcp(new Node(pl));
        bool callFillComplete = true;

        RCP<sparse_matrix_type> tA = reader_type::readSparseFile(fileName, comm, node, callFillComplete);

        if (tA.is_null())
          throw Exceptions::RuntimeError("The Tpetra::CrsMatrix returned from readSparseFile() is null.");

        RCP<Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > tmpA1 = rcp(new Xpetra::TpetraCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tA));
        RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >       tmpA2 = Teuchos::rcp_implicit_cast<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >(tmpA1);
        RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> >          A     = rcp(new Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA2));

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra GO=int enabled.");
#endif // HAVE_XPETRA_TPETRA_INST_INT_INT
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
      } else {
        throw Exceptions::RuntimeError("Xpetra:IO: you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
      }
    } else {
      // Custom file format (binary)
      std::ifstream ifs(fileName.c_str(), std::ios::binary);
      TEUCHOS_TEST_FOR_EXCEPTION(!ifs.good(), Exceptions::RuntimeError, "Can not read \"" << fileName << "\"");
      int m, n, nnz;
      ifs.read(reinterpret_cast<char*>(&m),   sizeof(m));
      ifs.read(reinterpret_cast<char*>(&n),   sizeof(n));
      ifs.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

      int myRank = comm->getRank();

      GlobalOrdinal indexBase = 0;
      RCP<Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >    rowMap = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, m, (myRank == 0 ? m : 0), indexBase, comm), rangeMap  = rowMap;
      RCP<Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >    colMap = Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(lib, n, (myRank == 0 ? n : 0), indexBase, comm), domainMap = colMap;
      RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > A   = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(rowMap, colMap, 1);

      TEUCHOS_TEST_FOR_EXCEPTION(sizeof(int) != sizeof(GlobalOrdinal), Exceptions::RuntimeError, "Incompatible sizes");

      if (myRank == 0) {
        Teuchos::Array<GlobalOrdinal> inds;
        Teuchos::Array<Scalar> vals;
        for (int i = 0; i < m; i++) {
          int row, rownnz;
          ifs.read(reinterpret_cast<char*>(&row),    sizeof(row));
          ifs.read(reinterpret_cast<char*>(&rownnz), sizeof(rownnz));
          inds.resize(rownnz);
          vals.resize(rownnz);
          for (int j = 0; j < rownnz; j++) {
            int index;
            ifs.read(reinterpret_cast<char*>(&index), sizeof(index));
            inds[j] = Teuchos::as<GlobalOrdinal>(index);
          }
          for (int j = 0; j < rownnz; j++) {
            double value;
            ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
            vals[j] = Teuchos::as<Scalar>(value);
          }
          A->insertGlobalValues(row, inds, vals);
        }
      }

      A->fillComplete(domainMap, rangeMap);

      return A;
    }


  } //Read()


  /*! @brief Read matrix from file in Matrix Market or binary format.

        If only rowMap is specified, then it is used for the domainMap and rangeMap, as well.
   */
  static Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> > Read(const std::string&   filename,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > rowMap,
      RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > colMap           = Teuchos::null,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > domainMap        = Teuchos::null,
      const RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > rangeMap         = Teuchos::null,
      const bool           callFillComplete = true,
      const bool           binary           = false,
      const bool           tolerant         = false,
      const bool           debug            = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(rowMap.is_null(), Exceptions::RuntimeError, "Utils::Read() : rowMap cannot be null");

    RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > domain = (domainMap.is_null() ? rowMap : domainMap);
    RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > range  = (rangeMap .is_null() ? rowMap : rangeMap);

    const Xpetra::UnderlyingLib lib = rowMap->lib();
    if (binary == false) {
      if (lib == Xpetra::UseEpetra) {
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
        Epetra_CrsMatrix *eA;
        const RCP<const Epetra_Comm> epcomm = Xpetra::toEpetra(rowMap->getComm());
        const Epetra_Map& epetraRowMap    = Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*rowMap);
        const Epetra_Map& epetraDomainMap = (domainMap.is_null() ? epetraRowMap : Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*domainMap));
        const Epetra_Map& epetraRangeMap  = (rangeMap .is_null() ? epetraRowMap : Xpetra::IO<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Map2EpetraMap(*rangeMap));
        int rv;
        if (colMap.is_null()) {
          rv = EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), epetraRowMap, epetraRangeMap, epetraDomainMap, eA);

        } else {
          const Epetra_Map& epetraColMap  = Map2EpetraMap(*colMap);
          rv = EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), epetraRowMap, epetraColMap, epetraRangeMap, epetraDomainMap, eA);
        }

        if (rv != 0)
          throw Exceptions::RuntimeError("EpetraExt::MatrixMarketFileToCrsMatrix return value of " + Teuchos::toString(rv));

        RCP<Epetra_CrsMatrix> tmpA = rcp(eA);
        RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >           A    = Convert_Epetra_CrsMatrix_ToXpetra_CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>(tmpA);

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
      } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
        typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
        typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>             reader_type;
        typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>               map_type;

        const RCP<const map_type> tpetraRowMap    = Map2TpetraMap(*rowMap);
        RCP<const map_type>       tpetraColMap    = (colMap.is_null()    ? Teuchos::null : Map2TpetraMap(*colMap));
        const RCP<const map_type> tpetraRangeMap  = (rangeMap.is_null()  ? tpetraRowMap  : Map2TpetraMap(*rangeMap));
        const RCP<const map_type> tpetraDomainMap = (domainMap.is_null() ? tpetraRowMap  : Map2TpetraMap(*domainMap));

        RCP<sparse_matrix_type> tA = reader_type::readSparseFile(filename, tpetraRowMap, tpetraColMap, tpetraDomainMap, tpetraRangeMap,
            callFillComplete, tolerant, debug);
        if (tA.is_null())
          throw Exceptions::RuntimeError("The Tpetra::CrsMatrix returned from readSparseFile() is null.");

        RCP<Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > tmpA1 = rcp(new Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tA));
        RCP<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >       tmpA2 = Teuchos::rcp_implicit_cast<Xpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >(tmpA1);
        RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> >          A     = rcp(new Xpetra::CrsMatrixWrap<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tmpA2));

        return A;
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra GO=int support.");
#endif
#else
        throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
      } else {
        throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
      }
    } else {
      // Custom file format (binary)
      std::ifstream ifs(filename.c_str(), std::ios::binary);
      TEUCHOS_TEST_FOR_EXCEPTION(!ifs.good(), Exceptions::RuntimeError, "Can not read \"" << filename << "\"");
      int m, n, nnz;
      ifs.read(reinterpret_cast<char*>(&m),   sizeof(m));
      ifs.read(reinterpret_cast<char*>(&n),   sizeof(n));
      ifs.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

      RCP<Xpetra::Matrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > A = Xpetra::MatrixFactory<Scalar,LocalOrdinal,GlobalOrdinal,Node>::Build(rowMap, colMap, 1);

      TEUCHOS_TEST_FOR_EXCEPTION(sizeof(int) != sizeof(GlobalOrdinal), Exceptions::RuntimeError, "Incompatible sizes");

      Teuchos::ArrayView<const GlobalOrdinal> rowElements = rowMap->getNodeElementList();
      Teuchos::ArrayView<const GlobalOrdinal> colElements = colMap->getNodeElementList();

      Teuchos::Array<GlobalOrdinal> inds;
      Teuchos::Array<Scalar> vals;
      for (int i = 0; i < m; i++) {
        int row, rownnz;
        ifs.read(reinterpret_cast<char*>(&row),    sizeof(row));
        ifs.read(reinterpret_cast<char*>(&rownnz), sizeof(rownnz));
        inds.resize(rownnz);
        vals.resize(rownnz);
        for (int j = 0; j < rownnz; j++) {
          int index;
          ifs.read(reinterpret_cast<char*>(&index), sizeof(index));
          inds[j] = colElements[Teuchos::as<LocalOrdinal>(index)];
        }
        for (int j = 0; j < rownnz; j++) {
          double value;
          ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
          vals[j] = Teuchos::as<Scalar>(value);
        }
        A->insertGlobalValues(rowElements[row], inds, vals);
      }
      A->fillComplete(domainMap, rangeMap);
      return A;
    }

  }
  //@}


  static RCP<Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> > ReadMultiVector (const std::string& fileName, const RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >& map) {
    Xpetra::UnderlyingLib lib = map->lib();

    if (lib == Xpetra::UseEpetra) {
      // taw: Oct 9 2015: do we need a specialization for <double,int,int>??
      //TEUCHOS_TEST_FOR_EXCEPTION(true, ::Xpetra::Exceptions::BadCast, "Epetra can only be used with Scalar=double and Ordinal=int");
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
      Epetra_MultiVector * MV;
      EpetraExt::MatrixMarketFileToMultiVector(fileName.c_str(), toEpetra(map), MV);
      return Xpetra::toXpetra<int,Node>(rcp(MV));
#else
      throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
    } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
      typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
      typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>                          reader_type;
      typedef Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>                            map_type;
      typedef Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>            multivector_type;

      RCP<const map_type>   temp = toTpetra(map);
      RCP<multivector_type> TMV  = reader_type::readDenseFile(fileName,map->getComm(),map->getNode(),temp);
      RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> >      rmv  = Xpetra::toXpetra(TMV);
      return rmv;
#else
  throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra GO=int support.");
#endif
#else
  throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
    } else {
      throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
    }


  }


  static RCP<const Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >   ReadMap         (const std::string& fileName, Xpetra::UnderlyingLib lib, const RCP<const Teuchos::Comm<int> >& comm) {
    if (lib == Xpetra::UseEpetra) {
      // do we need another specialization for <double,int,int> ??
      //TEUCHOS_TEST_FOR_EXCEPTION(true, ::Xpetra::Exceptions::BadCast, "Epetra can only be used with Scalar=double and Ordinal=int");
#if defined(HAVE_XPETRA_EPETRA) && defined(HAVE_XPETRA_EPETRAEXT)
      Epetra_Map *eMap;
      int rv = EpetraExt::MatrixMarketFileToMap(fileName.c_str(), *(Xpetra::toEpetra(comm)), eMap);
      if (rv != 0)
        throw Exceptions::RuntimeError("Error reading matrix with EpetraExt::MatrixMarketToMap (returned " + Teuchos::toString(rv) + ")");

      RCP<Epetra_Map> eMap1 = rcp(new Epetra_Map(*eMap));
      return Xpetra::toXpetra<int,Node>(*eMap1);
#else
      throw Exceptions::RuntimeError("Xpetra has not been compiled with Epetra and EpetraExt support.");
#endif
    } else if (lib == Xpetra::UseTpetra) {
#ifdef HAVE_XPETRA_TPETRA
#ifdef HAVE_XPETRA_TPETRA_INST_INT_INT
      typedef Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node> sparse_matrix_type;
      typedef Tpetra::MatrixMarket::Reader<sparse_matrix_type>                          reader_type;

      RCP<Node> node = rcp(new Node());

      RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > tMap = reader_type::readMapFile(fileName, comm, node);
      if (tMap.is_null())
        throw Exceptions::RuntimeError("The Tpetra::Map returned from readSparseFile() is null.");

      return Xpetra::toXpetra(tMap);
#else
      throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra GO=int support.");
#endif
#else
      throw Exceptions::RuntimeError("Xpetra has not been compiled with Tpetra support.");
#endif
    } else {
      throw Exceptions::RuntimeError("Utils::Read : you must specify Xpetra::UseEpetra or Xpetra::UseTpetra.");
    }

  }


};


} // end namespace Xpetra

#define XPETRA_IO_SHORT

#endif /* PACKAGES_XPETRA_SUP_UTILS_XPETRA_IO_HPP_ */
