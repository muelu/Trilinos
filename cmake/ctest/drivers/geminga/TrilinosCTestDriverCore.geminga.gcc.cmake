# @HEADER
# ************************************************************************
#
#            Trilinos: An Object-Oriented Solver Framework
#                 Copyright (2001) Sandia Corporation
#
#
# Copyright (2001) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000, there is a non-exclusive license for use of this
# work by or on behalf of the U.S. Government.  Export of this program
# may require a license from the United States Government.
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# NOTICE:  The United States Government is granted for itself and others
# acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
# license in this data to reproduce, prepare derivative works, and
# perform publicly and display publicly.  Beginning five (5) years from
# July 25, 2001, the United States Government is granted for itself and
# others acting on its behalf a paid-up, nonexclusive, irrevocable
# worldwide license in this data to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.
#
# NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT
# OF ENERGY, NOR SANDIA CORPORATION, NOR ANY OF THEIR EMPLOYEES, MAKES
# ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
# RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
# INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
# THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
#
# ************************************************************************
# @HEADER


INCLUDE("${CTEST_SCRIPT_DIRECTORY}/../../TrilinosCTestDriverCore.cmake")

#
# Platform/compiler specific options for geminga using gcc
#

MACRO(TRILINOS_SYSTEM_SPECIFIC_CTEST_DRIVER)

  # Base of Trilinos/cmake/ctest then BUILD_DIR_NAME

  IF(COMM_TYPE STREQUAL MPI)
    string(TOUPPER $ENV{SEMS_MPI_NAME} UC_MPI_NAME)
    SET(BUILD_DIR_NAME ${UC_MPI_NAME}-$ENV{SEMS_MPI_VERSION}_${BUILD_TYPE}_${BUILD_NAME_DETAILS})
  ELSE()
    SET(BUILD_DIR_NAME ${COMM_TYPE}-${BUILD_TYPE}_${BUILD_NAME_DETAILS})
  ENDIF()

  SET(Trilinos_REPOSITORY_LOCATION_NIGHTLY_DEFAULT "git@github.com:muelu/Trilinos.git")

  SET(CTEST_DASHBOARD_ROOT  "${TRILINOS_CMAKE_DIR}/../../${BUILD_DIR_NAME}" )
  SET(CTEST_NOTES_FILES     "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}" )
  SET(CTEST_BUILD_FLAGS     "-j35 -i" )

  SET_DEFAULT(CTEST_PARALLEL_LEVEL                  "35" )
  SET_DEFAULT(Trilinos_ENABLE_SECONDARY_TESTED_CODE ON)
  SET(Trilinos_CTEST_DO_ALL_AT_ONCE FALSE)
  SET_DEFAULT(Trilinos_EXCLUDE_PACKAGES             ${EXTRA_EXCLUDE_PACKAGES} TriKota Optika)

  SET(EXTRA_SYSTEM_CONFIGURE_OPTIONS
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_VERBOSE_MAKEFILE=ON"

    "-DTrilinos_ENABLE_COMPLEX:BOOL=OFF"

    "-DTrilinos_ENABLE_Fortran=OFF"

    "-DSuperLU_INCLUDE_DIRS=$ENV{SEMS_SUPERLU_INCLUDE_PATH}"
    "-DSuperLU_LIBRARY_DIRS=$ENV{SEMS_SUPERLU_LIBRARY_PATH}"
    "-DSuperLU_LIBRARY_NAMES=superlu"

    "-DBoost_INCLUDE_DIRS:STRING=$ENV{SEMS_BOOST_INCLUDE_PATH}"
    "-DBoost_LIBRARY_DIRS:STRING=$ENV{SEMS_BOOST_LIBRARY_PATH}"
    "-DBoostLib_INCLUDE_DIRS:STRING=$ENV{SEMS_BOOST_INCLUDE_PATH}"
    "-DBoostLib_LIBRARY_DIRS:STRING=$ENV{SEMS_BOOST_LIBRARY_PATH}"

    "-DNetcdf_LIBRARY_DIRS:STRING=$ENV{SEMS_NETCDF_LIBRARY_PATH}"
    "-DNetcdf_INCLUDE_DIRS:STRING=$ENV{SEMS_NETCDF_INCLUDE_PATH}"

    ### PACKAGE CONFIGURATION ###

    ### MISC ###
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
    )

  SET_DEFAULT(COMPILER_VERSION "$ENV{SEMS_COMPILER_NAME}-$ENV{SEMS_COMPILER_VERSION}")

  # Options for valgrind, if needed
  SET(CTEST_MEMORYCHECK_COMMAND_OPTIONS
      "--trace-children=yes --leak-check=full --gen-suppressions=all --error-limit=no" ${CTEST_MEMORYCHECK_COMMAND_OPTIONS} )
  SET(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_SCRIPT_DIRECTORY}/valgrind_suppressions.txt")


  # Ensure that MPI is on for all parallel builds that might be run.
  IF(COMM_TYPE STREQUAL MPI)

    SET(EXTRA_SYSTEM_CONFIGURE_OPTIONS
        ${EXTRA_SYSTEM_CONFIGURE_OPTIONS}
        "-DTPL_ENABLE_MPI=ON"
            "-DMPI_BASE_DIR:PATH=$ENV{SEMS_OPENMPI_ROOT}"
            "-DMPI_EXEC_POST_NUMPROCS_FLAGS:STRING=--bind-to\\\;socket\\\;--map-by\\\;socket"
       )


  ELSE()

    SET( EXTRA_SYSTEM_CONFIGURE_OPTIONS
      ${EXTRA_SYSTEM_CONFIGURE_OPTIONS}
      "-DCMAKE_CXX_COMPILER=$ENV{SEMS_COMPILER_ROOT}/bin/g++"
      "-DCMAKE_C_COMPILER=$ENV{SEMS_COMPILER_ROOT}/bin/gcc"
      )

  ENDIF()

  TRILINOS_CTEST_DRIVER()

ENDMACRO()
