TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ObjectiveConstraintConversion
  SOURCES test_01.cpp
  ARGS PrintItAll
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "TEST PASSED"
  ADD_DIR_TO_NAME
  )


TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ScalarLinearConstraint
  SOURCES test_02.cpp
  ARGS PrintItAll
  NUM_MPI_PROCS 1
  PASS_REGULAR_EXPRESSION "TEST PASSED"
  ADD_DIR_TO_NAME
  )

# Moved to adapters/mpi/test/function
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  Constraint_TimeSimOpt
#  SOURCES test_03.cpp
#  ARGS PrintItAll
#  NUM_MPI_PROCS 1
#  PASS_REGULAR_EXPRESSION "TEST PASSED"
#  ADD_DIR_TO_NAME
#  )

# Moved to adapters/mpi/test/function
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  Constraint_PinTSimOpt 
#  SOURCES test_04.cpp
#  ARGS PrintItAll
#  COMM mpi
#  NUM_MPI_PROCS 3
#  PASS_REGULAR_EXPRESSION "TEST PASSED"
#  ADD_DIR_TO_NAME
#  )
