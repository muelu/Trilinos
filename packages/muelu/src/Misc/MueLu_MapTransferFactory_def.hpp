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

#ifndef MUELU_MAPTRANSFERFACTORY_DEF_HPP_
#define MUELU_MAPTRANSFERFACTORY_DEF_HPP_

#include "MueLu_MapTransferFactory_decl.hpp"

#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_Matrix.hpp>

#include "MueLu_Level.hpp"
#include "MueLu_FactoryManagerBase.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  MapTransferFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::MapTransferFactory() { }


  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> MapTransferFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set< std::string >           ("Map name",               "", "Name of the map to be transferred");
    validParamList->set< RCP<const FactoryBase> >("Map factory", Teuchos::null, "Generating factory of the map to be transferred");
    validParamList->set< RCP<const FactoryBase> >("P",           Teuchos::null, "Tentative prolongator factory");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MapTransferFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& fineLevel, Level& coarseLevel) const {
    const ParameterList& pL = GetParameterList();
    std::string mapName = pL.get<std::string>("Map name");
    Input(fineLevel, mapName, "Map factory");

    Input(coarseLevel, "P", "Ptent");
    // request Ptent
    // note that "P" provided by the user (through XML file) is supposed to be of type TentativePFactory
    Teuchos::RCP<const FactoryBase> tentPFact = GetFactory("P");
    if (tentPFact == Teuchos::null)
      tentPFact = coarseLevel.GetFactoryManager()->GetFactory("Ptent");
    coarseLevel.DeclareInput("P", tentPFact.get(), this);
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void MapTransferFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& fineLevel, Level& coarseLevel) const {
    Monitor m(*this, "Contact Map transfer factory");

    const ParameterList & pL = GetParameterList();
    std::string mapName     = pL.get<std::string>("map: name");

    if (fineLevel.IsAvailable(mapName, mapFact_.get())==false)
      GetOStream(Runtime0) << "MapTransferFactory::Build: User provided map " << mapName << " not found in Level class." << std::endl;

    // fetch map extractor from level
    RCP<const Map> transferMap = fineLevel.Get<RCP<const Map> >(mapName,mapFact_.get());

    // Get default tentative prolongator factory
    // Getting it that way ensure that the same factory instance will be used for both SaPFactory and NullspaceFactory.
    // -- Warning: Do not use directly initialPFact_. Use initialPFact instead everywhere!
    RCP<const FactoryBase> tentPFact = GetFactory("P");
    if (tentPFact == Teuchos::null)
      tentPFact = coarseLevel.GetFactoryManager()->GetFactory("Ptent");
    TEUCHOS_TEST_FOR_EXCEPTION(!coarseLevel.IsAvailable("P", tentPFact.get()), Exceptions::RuntimeError,
                               "MueLu::MapTransferFactory::Build(): P (generated by TentativePFactory) not available.");
    RCP<Matrix> Ptent = coarseLevel.Get<RCP<Matrix> >("P", tentPFact.get());

    std::vector<GO > coarseMapGids;

    // loop over local rows of Ptent
    for (size_t row=0; row<Ptent->getNodeNumRows(); row++) {
      GO grid = Ptent->getRowMap()->getGlobalElement(row);

      if (transferMap->isNodeGlobalElement(grid)) {
        Teuchos::ArrayView<const LO> indices;
        Teuchos::ArrayView<const SC> vals;
        Ptent->getLocalRowView(row, indices, vals);

        for (size_t i = 0; i < as<size_t>(indices.size()); i++) {
          // mark all columns in Ptent(grid,*) to be coarse Dofs of next level transferMap
          GO gcid = Ptent->getColMap()->getGlobalElement(indices[i]);
          coarseMapGids.push_back(gcid);
        }
      } // end if isNodeGlobalElement(grid)
    }

    // build column maps
    const GO INVALID = Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid();
    std::sort(coarseMapGids.begin(), coarseMapGids.end());
    coarseMapGids.erase(std::unique(coarseMapGids.begin(), coarseMapGids.end()), coarseMapGids.end());
    Teuchos::ArrayView<GO> coarseMapGidsView(&coarseMapGids[0], coarseMapGids.size());
    Teuchos::RCP<const Map> coarseTransferMap = MapFactory::Build(Ptent->getColMap()->lib(), INVALID, coarseMapGidsView, Ptent->getColMap()->getIndexBase(), Ptent->getColMap()->getComm());

    // store map extractor in coarse level
    coarseLevel.Set(mapName, coarseTransferMap, mapFact_.get());
  }

} // namespace MueLu

#endif /* MUELU_MAPTRANSFERFACTORY_DEF_HPP_ */
