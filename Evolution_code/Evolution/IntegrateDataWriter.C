/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2013 Lawrence Livermore National Security, LLC
 * Description:   Writes data files for visualization by VisIt
 *
 ************************************************************************/

#ifndef included_appu_IntegrateDataWriter_C
#define included_appu_IntegrateDataWriter_C

#include "IntegrateDataWriter.h"
#include <iostream>

#include "SAMRAI/tbox/TimerManager.h"
#include "SAMRAI/hier/BoxLevelConnectorUtils.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/BoxLevel.h"
#include "SAMRAI/hier/RealBoxConstIterator.h"
#include "SAMRAI/hier/VariableDatabase.h"
#include "SAMRAI/pdat/CellDataFactory.h"
#include "SAMRAI/pdat/NodeDataFactory.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

#define vector2D(v, i, j) (v)[i+ilast*(j)]
#define vectorN2D(v, i, j) (v)[i+ilastN*(j)]

#define vector3D(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]
#define vectorN3D(v, i, j, k) (v)[i+ilastN*(j+jlastN*(k))]

const float IntegrateDataWriter::INTEGRAL_DATAWRITER_VERSION_NUMBER = 2.0;
const int IntegrateDataWriter::INTEGRAL_NAME_BUFSIZE = 128;
const int IntegrateDataWriter::INTEGRAL_UNDEFINED_INDEX = -1;
const int IntegrateDataWriter::INTEGRAL_MASTER = 0;
int IntegrateDataWriter::d_mask_id;
double IntegrateDataWriter::d_mask_simulation_time;

tbox::StartupShutdownManager::Handler
IntegrateDataWriter::s_initialize_handler(IntegrateDataWriter::initializeCallback, 0, 0, IntegrateDataWriter::finalizeCallback, tbox::StartupShutdownManager::priorityTimers);

std::shared_ptr<tbox::Timer> IntegrateDataWriter::t_write_plot_data;

using namespace std;

/*
 *************************************************************************
 *
 * The constructor --- sets default object state.
 *
 *************************************************************************
 */

IntegrateDataWriter::IntegrateDataWriter(
   const std::vector<std::string> calculation,
   const std::string& object_name,
   const std::string& dump_directory_name,
   int number_procs_per_file,
   bool is_multiblock):
   d_calculation(calculation.begin(), calculation.end())
{
   TBOX_ASSERT(!object_name.empty());
   TBOX_ASSERT(number_procs_per_file > 0);

   d_object_name = object_name;

   d_time_step_number = INTEGRAL_UNDEFINED_INDEX;
   d_top_level_directory_name = dump_directory_name;
   d_mask_simulation_time = -1;
}

/*
 *************************************************************************
 *
 * The destructor implicitly deallocates the list of plot data items.
 *
 *************************************************************************
 */

IntegrateDataWriter::~IntegrateDataWriter()
{
}

/*
 *************************************************************************
 *
 * Register (non-derived) plot quantities: scalar, vector or tensor.
 *
 *************************************************************************
 */
void
IntegrateDataWriter::registerPlotQuantity(
   const std::string& variable_name,
   const std::string& variable_type,
   const int patch_data_index,
   const int start_depth_index,
   const double scale_factor,
   const std::string& variable_centering)
{

   TBOX_ASSERT(!variable_name.empty());
   TBOX_ASSERT(!variable_type.empty());
   TBOX_ASSERT(patch_data_index >= -1);
   TBOX_ASSERT(start_depth_index >= 0);

   /*
    * Check for name conflicts with existing registered variables.
    */
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
        ipi != d_plot_items.end(); ipi++) {
      if (ipi->d_var_name == variable_name) {
         TBOX_ERROR("IntegrateDataWriter::registerPlotQuantity()"
            << "\n    Attempting to register variable with name "
            << variable_name << "\n    more than once." << std::endl);
      }
   }

   /*
    * Create a plot item and initialize its characteristics.
    */
   VisItItem plotitem;

   initializePlotItem(plotitem,
      variable_name,
      variable_type,
      patch_data_index,
      start_depth_index,
      scale_factor,
      variable_centering);

   d_plot_items.push_back(plotitem);

}

/*
 *************************************************************************
 *
 * Reset previously-registered scalar/vector variable to new data id
 * and depth index on the given level.  This allows the use of different
 * patch data ids for the same quantity on different hierarchy levels.
 * We check to make sure that the factory at the given index is
 * defined and consistent with the original registration.
 *
 *************************************************************************
 */

void
IntegrateDataWriter::resetLevelPlotQuantity(
   const std::string& variable_name,
   const int level_number,
   const int patch_data_index,
   const int start_depth_index)
{
   TBOX_ASSERT(!variable_name.empty());
   TBOX_ASSERT(level_number >= 0);
   TBOX_ASSERT(patch_data_index >= -1);
   TBOX_ASSERT(start_depth_index >= 0);

   /*
    * Verify the supplied patch data index has the same type and centering
    * as the plot item its replacing.
    */
   std::shared_ptr<hier::PatchDataFactory> factory(
      hier::VariableDatabase::getDatabase()->
      getPatchDescriptor()->
      getPatchDataFactory(patch_data_index));

   bool found_type = false;
   variable_data_type vdt = INTEGRAL_DATA_TYPE_BAD;
   variable_centering vc = INTEGRAL_CENTERING_BAD;

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));

      if (ffactory) {
         vdt = INTEGRAL_FLOAT;
         vc = INTEGRAL_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
      if (ffactory) {
         vdt = INTEGRAL_FLOAT;
         vc = INTEGRAL_NODE;
         found_type = true;
      }
   }

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = INTEGRAL_DOUBLE;
         vc = INTEGRAL_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = INTEGRAL_DOUBLE;
         vc = INTEGRAL_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = INTEGRAL_INT;
         vc = INTEGRAL_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = INTEGRAL_INT;
         vc = INTEGRAL_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      TBOX_ERROR("IntegrateDataWriter::resetLevelPlotQuantity()"
         << "\n     Unable to determine type and centering"
         << "\n     of supplied patch data index."
         << "\n     ***Exiting" << std::endl);
   }

   /*
    * Find variable in the list of maintained vars.
    */
   bool found_var = false;
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
        (!found_var && ipi != d_plot_items.end()); ipi++) {
      if (ipi->d_var_name == variable_name) {

         /*
          * Check to make sure supplied variable has same type and
          * centering as registered one.
          */
         if ((ipi->d_var_data_type != vdt) ||
             (ipi->d_var_centering != vc)) {
            TBOX_ERROR("IntegrateDataWriter::resetLevelPlotQuantity()"
               << "\n     The supplied patch data id has a different"
               << "\n     type and centering from the one originally"
               << "\n     registered.  hier::Variable name: "
               << variable_name
               << "\n     ***Exiting" << std::endl);
         }
      }
   }

   if (!found_var) {
      TBOX_ERROR("IntegrateDataWriter::resetLevelPlotQuantity()"
         << "\n     Could not find the variable: "
         << variable_name
         << "\n     in the list of registered plot items."
         << "\n     Be sure registerPlotQuantity() has been"
         << "\n     called for the variable."
         << "\n     ***Exiting" << std::endl);
   }

}

/*
 *************************************************************************
 *
 * Private method which initializes a VisIt variable based on user
 * input.
 *
 *************************************************************************
 */
void
IntegrateDataWriter::initializePlotItem(
   VisItItem& plotitem,
   const std::string& variable_name,
   const std::string& variable_type,
   const int patch_data_index,
   const int start_depth_index,
   const double scale_factor,
   const std::string& variable_centering)
{
   TBOX_ASSERT(!variable_name.empty());
   TBOX_ASSERT(!variable_type.empty());
   TBOX_ASSERT(patch_data_index >= -1);
   TBOX_ASSERT(start_depth_index >= 0);

   plotitem.d_var_name = variable_name;

   /*
    * Set variable type.
    */
   if (variable_type == "SCALAR") {
      plotitem.d_var_type = INTEGRAL_SCALAR;
      plotitem.d_depth = 1;
   } else {
      TBOX_ERROR("IntegrateDataWriter::registerPlotItem"
         << "\n    variable_type " << variable_type
         << "\n    is unsupported.  You must use SCALAR.  Exiting***" << std::endl);
   }

   /*
    * Set variable centering.  If the variable supplied a valid patch
    * index, we check the factory of the variable first to try to
    * determine centering from that.  If we cannot determine
    * the type, we use the "variable_centering" provided by the user.
    * Note that we only set the centering for variables that are not
    * derived, materials, or species, since these variables do not have
    * a supplied patch data index because they are packed by the user.
    */
   bool found_type = false;
   int var_depth = 0;
   if (patch_data_index >= 0) {

      std::shared_ptr<hier::PatchDataFactory> factory(
         hier::VariableDatabase::getDatabase()->
         getPatchDescriptor()->
         getPatchDataFactory(patch_data_index));
      if (!factory) {
         TBOX_ERROR("IntegrateDataWriter::registerPlotItem"
            << "\n    patch data array index = " << patch_data_index
            << "\n    for variable = " << variable_name
            << "\n    is invalid" << std::endl);
      } else {

         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = INTEGRAL_CELL;
               plotitem.d_var_data_type = INTEGRAL_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = INTEGRAL_CELL;
               plotitem.d_var_data_type = INTEGRAL_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = INTEGRAL_CELL;
               plotitem.d_var_data_type = INTEGRAL_INT;
               var_depth = ifactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = INTEGRAL_NODE;
               plotitem.d_var_data_type = INTEGRAL_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = INTEGRAL_NODE;
               plotitem.d_var_data_type = INTEGRAL_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = INTEGRAL_NODE;
               plotitem.d_var_data_type = INTEGRAL_INT;
               var_depth = ifactory->getDepth();
               found_type = true;
            }
         }

         /*
          * Make sure variable depth is sufficient for the specified type
          * (SCALAR/VECTOR/TENSOR) with the start depth index.
          */
         int end_depth = start_depth_index + plotitem.d_depth;
         if (var_depth < end_depth) {
            TBOX_ERROR("IntegrateDataWriter::registerPlotItem"
               << "\n    The variable: " << variable_name
               << "\n    has insufficient depth for the type"
               << "\n    and start depth index registered."
               << "\n      var_type:           " << variable_type
               << "\n      start_depth_index:  " << start_depth_index
               << "\n      required min depth: " << end_depth
               << std::endl);
         }

      } // factory is Null

   } // valid patch data index

   if (!found_type) {
      if (variable_centering == "CELL") {
         plotitem.d_var_centering = INTEGRAL_UNKNOWN_CELL;
      } else if (variable_centering == "NODE") {
         plotitem.d_var_centering = INTEGRAL_UNKNOWN_NODE;
      } else {
         TBOX_ERROR("IntegrateDataWriter::registerPlotItem"
            << "\n     Unable to determine the centering for"
            << "\n     this variable; it must be supplied."
            << "\n     The variable_centering argument is "
            << variable_centering
            << "\n     Possible entries are CELL or NODE"
            << "\n     ***Exiting" << std::endl);
      }

      /*
       * When the var centering is specified by the user, we assume
       * it is of type double.  This implies the user should NOT try
       * to register their undefined variables with type float or int.
       */
      plotitem.d_var_data_type = INTEGRAL_DOUBLE;
   }
   plotitem.d_patch_data_index = patch_data_index;
   plotitem.d_visit_var_name.resize(plotitem.d_depth);
   plotitem.d_integral_value.resize(plotitem.d_depth);
   plotitem.d_l2norm_value.resize(plotitem.d_depth);
   plotitem.d_min_value.resize(plotitem.d_depth);
   plotitem.d_max_value.resize(plotitem.d_depth);
   plotitem.d_absmin_value.resize(plotitem.d_depth);
   plotitem.d_absmax_value.resize(plotitem.d_depth);

   char temp_buf[INTEGRAL_NAME_BUFSIZE];
   for (int i = 0; i < plotitem.d_depth; i++) {
      if (plotitem.d_depth == 1) {
         plotitem.d_visit_var_name[i] = variable_name;
      } else {
         sprintf(temp_buf, ".%02d", i);
         plotitem.d_visit_var_name[i] = variable_name + temp_buf;
      }
   }
}

/*
 *************************************************************************
 *
 * Write plot data from given hierarchy to HDF file
 *
 *************************************************************************
 */

void
IntegrateDataWriter::writePlotData(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   int time_step_number,
   double simulation_time)
{
   TBOX_ASSERT(hierarchy);
   TBOX_ASSERT(time_step_number >= 0);
   TBOX_ASSERT(!d_top_level_directory_name.empty());

   /*
    * Currently, this class does not work unless the nodes
    * have globally sequentialized indices.  Check for these.
    */
   hier::BoxLevelConnectorUtils dlbg_edge_utils;

   for (int ln = 0; ln < hierarchy->getNumberOfLevels(); ++ln) {
      const hier::BoxLevel& unsorted_box_level =
         *hierarchy->getPatchLevel(ln)->getBoxLevel();
      std::shared_ptr<hier::BoxLevel> sorted_box_level;
      std::shared_ptr<hier::MappingConnector> unused_sorting_map;
      dlbg_edge_utils.makeSortingMap(
         sorted_box_level,
         unused_sorting_map,
         unsorted_box_level,
         false,
         true);
      if (*sorted_box_level != unsorted_box_level) {
         TBOX_ERROR(
            "IntegrateDataWriter: Encountered existing limitation of IntegrateDataWriter\n"
            << "This class cannot write files unless all patch levels have\n"
            << "globally sequentialized nodes.  This can be accomplished\n"
            << "by the sequentialize_patch_indices = TRUE input flag in\n"
            << "GriddingAlgorithm.  This problem can (and should\n"
            << "be fixed soon.");

      }
   }

   t_write_plot_data->start();

   if (time_step_number <= d_time_step_number) {
      TBOX_ERROR("IntegrateDataWriter::writePlotData"
         << "\n    data writer with name " << d_object_name
         << "\n    time step number: " << time_step_number
         << " is <= last time step number: " << d_time_step_number
         << std::endl);
   }
   d_time_step_number = time_step_number;

   if (d_top_level_directory_name.empty()) {
      TBOX_ERROR("IntegrateDataWriter::writePlotData"
         << "\n    data writer with name " << d_object_name
         << "\n     Dump Directory Name is not set" << std::endl);
   }

   writeFiles(hierarchy, simulation_time);

   t_write_plot_data->stop();
}


/*
 *************************************************************************
 *
 * Private function to coordinate writing HDF plot files.
 *
 *************************************************************************
 */

void
IntegrateDataWriter::writeFiles(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   double simulation_time)
{
   TBOX_ASSERT(hierarchy);

// Disable Intel warning about conversions
#ifdef __INTEL_COMPILER
#pragma warning (disable:810)
#endif

  //Create output directory
  const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
  int my_proc = mpi.getRank();
  tbox::Utilities::recursiveMkdir(d_top_level_directory_name);
  
  if (d_mask_simulation_time != simulation_time) {
    calculateMask(hierarchy);
    d_mask_simulation_time = simulation_time;
  }
  tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();

  clearValues();
  calculateReductions(hierarchy);

  for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
    for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
      std::string varname = d_top_level_directory_name + "/" + ipi->d_visit_var_name[depth_id];
      if(std::find(d_calculation.begin(), d_calculation.end(), "INTEGRAL") != d_calculation.end()) {
        //Calculate local integral
        double integral_value = ipi->d_integral_value[depth_id];
        mpi.AllReduce(&integral_value, 1, MPI_SUM);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_INTEGRAL";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" << integral_value << std::endl;
          outputfile.close();
        }
      }
      if(std::find(d_calculation.begin(), d_calculation.end(), "L2NORM") != d_calculation.end()) {
        //Calculate local integral
        double integral_value = ipi->d_l2norm_value[depth_id];
        mpi.AllReduce(&integral_value, 1, MPI_SUM);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_L2NORM";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" <<integral_value<< std::endl;
          outputfile.close();
        }
      }
      if(std::find(d_calculation.begin(), d_calculation.end(), "MAX") != d_calculation.end()) {
        //Calculate local integral
        double max_value = ipi->d_max_value[depth_id];
        mpi.AllReduce(&max_value, 1, MPI_MAX);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_MAX";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" << max_value << std::endl;
          outputfile.close();
        }
      }
      if(std::find(d_calculation.begin(), d_calculation.end(), "MIN") != d_calculation.end()) {
        //Calculate local integral
        double min_value = ipi->d_min_value[depth_id];
        mpi.AllReduce(&min_value, 1, MPI_MIN);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_MIN";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" << min_value << std::endl;
          outputfile.close();
        }
      }
      if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMAX") != d_calculation.end()) {
        //Calculate local integral
        double absmax_value = ipi->d_absmax_value[depth_id];
        mpi.AllReduce(&absmax_value, 1, MPI_MAX);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_ABSMAX";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" << absmax_value << std::endl;
          outputfile.close();
        }
      }
      if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
        //Calculate local integral
        double absmin_value = ipi->d_absmin_value[depth_id];
        mpi.AllReduce(&absmin_value, 1, MPI_MIN);
        // Write variable files
        if (my_proc == INTEGRAL_MASTER) {
          ofstream outputfile;
          char name[1024];
          std::string varname_int = varname + "_ABSMIN";
          strcpy(name, varname_int.c_str());
          // Create file if does not exist, otherwise open at the end of the file
          if (simulation_time == 0) {
            outputfile.open (name, ios::out);
          }
          else {
            outputfile.open (name, ios::out | ios::app);
          }
          outputfile << simulation_time << "\t" << absmin_value << std::endl;
          outputfile.close();
        }
      }
    }
  }
}

/*
 *************************************************************************
 *
 * Private function to calculate the mask in multi-level mesh
 *
 *************************************************************************
 */

void 
IntegrateDataWriter::calculateMask(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy)
{
   TBOX_ASSERT(hierarchy);
   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

   char temp_buf[INTEGRAL_NAME_BUFSIZE];
   std::shared_ptr<tbox::Database> level_HDFGroup, patch_HDFGroup;

   for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ln++) {
      std::shared_ptr<hier::PatchLevel> patch_level(hierarchy->getPatchLevel(ln));
      patch_level->getGlobalizedBoxLevel();
      for (hier::PatchLevel::iterator ip(patch_level->begin()); ip != patch_level->end(); ++ip) {
         const std::shared_ptr< hier::Patch >& patch = *ip;
         //Set mask active in all the patch
         ((pdat::CellData<int> *) patch->getPatchData(d_mask_id).get())->fill(1);
         if (ln <hierarchy->getFinestLevelNumber()) {
            std::shared_ptr<hier::PatchLevel> patch_levelG(hierarchy->getPatchLevel(ln + 1));
            const hier::BoxContainer& globalBoxes = patch_levelG->getGlobalizedBoxLevel().getGlobalBoxes();
            //Then check for interection with finer level boxes
            const hier::Box box = patch->getBox();
            for (hier::BoxContainer::const_iterator ibG(globalBoxes.begin()); ibG != globalBoxes.end(); ++ibG) {
               hier::Box boxG(*ibG);
               //Coarse finer box before intersect
               boxG.coarsen(patch_levelG->getRatioToCoarserLevel());
               //When there is intersection, set mask to 0 in the intersection cells
               if (box.intersects(boxG)) {
                  hier::Box intersection(box * boxG);
                  ((pdat::CellData<int> *) patch->getPatchData(d_mask_id).get())->fillAll(0, intersection);
               }
            }
         }
      }
   }

}

/*
 *************************************************************************
 *
 * Private function to perform the calculations
 *
 *************************************************************************
 */
void 
IntegrateDataWriter::calculateReductions(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy)
{
   if (hierarchy->getDim().getValue() == 2) {
     for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ln++) {
         std::shared_ptr<hier::PatchLevel> patch_level(hierarchy->getPatchLevel(ln));
         for (hier::PatchLevel::iterator ip(patch_level->begin()); ip != patch_level->end(); ++ip) {
            const std::shared_ptr< hier::Patch >& patch = *ip;
            int* mask = ((pdat::CellData<int> *) patch->getPatchData(d_mask_id).get())->getPointer();
            const hier::Index boxfirst = patch->getPatchData(d_mask_id)->getGhostBox().lower();
            const hier::Index boxlast  = patch->getPatchData(d_mask_id)->getGhostBox().upper();
            const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
            const double* dx  = patch_geom->getDx();

            int ilast = boxlast(0)-boxfirst(0) + 1;
            int jlast = boxlast(1)-boxfirst(1) + 1;

            int ilastN = boxlast(0)-boxfirst(0) + 2;
            int jlastN = boxlast(1)-boxfirst(1) + 2;

            const hier::IntVector ghost = patch->getPatchData(d_mask_id)->getGhostCellWidth();

            for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
               for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
                  double integral = 0;
                  double norm = 0;
                  double min = std::numeric_limits<double>::max();
                  double max = std::numeric_limits<double>::min();
                  double absmin = std::numeric_limits<double>::max();
                  double absmax = 0;
                  int patch_data_id = ipi->d_patch_data_index;
                  std::shared_ptr<const pdat::NodeData<double> > dpdata(SAMRAI_SHARED_PTR_CAST<const pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(patch_data_id)));
                  const double* dat_ptr = dpdata->getPointer(depth_id);
                  for(int j = ghost[1]; j < jlast - ghost[1]; j++) {
                     for(int i = ghost[0]; i < ilast - ghost[0]; i++) {
                        if (vector2D(mask, i, j) == 1) {
                           double calc = 0.25 * (vectorN2D(dat_ptr, i, j) + vectorN2D(dat_ptr, i, j + 1) + vectorN2D(dat_ptr, i + 1, j) + vectorN2D(dat_ptr, i + 1, j + 1));
                           if(std::find(d_calculation.begin(), d_calculation.end(), "INTEGRAL") != d_calculation.end()) {
                              integral = integral + dx[0]*dx[1] * calc;
                           }
                           if(std::find(d_calculation.begin(), d_calculation.end(), "L2NORM") != d_calculation.end()) {
                              norm = norm + dx[0]*dx[1] * calc*calc;
                           }
                           if(std::find(d_calculation.begin(), d_calculation.end(), "MIN") != d_calculation.end()) {
                              min = std::min(min, vectorN2D(dat_ptr, i, j));
                           }
                           if(std::find(d_calculation.begin(), d_calculation.end(), "MAX") != d_calculation.end()) {
                              max = std::max(max, vectorN2D(dat_ptr, i, j));
                           }
                           if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                              absmin = std::min(absmin, fabs(vectorN2D(dat_ptr, i, j)));
                           }
                           if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMAX") != d_calculation.end()) {
                              absmax = std::max(absmax, fabs(vectorN2D(dat_ptr, i, j)));
                           }
                        }
                     }
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "INTEGRAL") != d_calculation.end()) {
                     ipi->d_integral_value[depth_id] = ipi->d_integral_value[depth_id] + integral;
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "L2NORM") != d_calculation.end()) {
                     ipi->d_l2norm_value[depth_id] = ipi->d_l2norm_value[depth_id] + norm;
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "MIN") != d_calculation.end()) {
                     ipi->d_min_value[depth_id] = std::min(min, ipi->d_min_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "MAX") != d_calculation.end()) {
                     ipi->d_max_value[depth_id] = std::max(max, ipi->d_max_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                     ipi->d_absmin_value[depth_id] = std::min(absmin, ipi->d_absmin_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                     ipi->d_absmax_value[depth_id] = std::max(absmax, ipi->d_absmax_value[depth_id]);
                  }
               }
            }
         }
      }
   } else if (hierarchy->getDim().getValue() == 3) {
      for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ln++) {
         std::shared_ptr<hier::PatchLevel> patch_level(hierarchy->getPatchLevel(ln));
         for (hier::PatchLevel::iterator ip(patch_level->begin()); ip != patch_level->end(); ++ip) {
            const std::shared_ptr< hier::Patch >& patch = *ip;
            int* mask = ((pdat::CellData<int> *) patch->getPatchData(d_mask_id).get())->getPointer();
            const hier::Index boxfirst = patch->getPatchData(d_mask_id)->getGhostBox().lower();
            const hier::Index boxlast  = patch->getPatchData(d_mask_id)->getGhostBox().upper();
            const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
            const double* dx  = patch_geom->getDx();

            int ilast = boxlast(0)-boxfirst(0) + 1;
            int jlast = boxlast(1)-boxfirst(1) + 1;
            int klast = boxlast(2)-boxfirst(2) + 1;

            int ilastN = boxlast(0)-boxfirst(0) + 2;
            int jlastN = boxlast(1)-boxfirst(1) + 2;
            int klastN = boxlast(2)-boxfirst(2) + 2;

            const hier::IntVector ghost = patch->getPatchData(d_mask_id)->getGhostCellWidth();

            for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
               for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
                  double integral = 0;
                  double norm = 0;
                  double min = std::numeric_limits<double>::max();
                  double max = std::numeric_limits<double>::min();
                  double absmin = std::numeric_limits<double>::max();
                  double absmax = 0;
                  int patch_data_id = ipi->d_patch_data_index;
                  std::shared_ptr<const pdat::NodeData<double> > dpdata(SAMRAI_SHARED_PTR_CAST<const pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(patch_data_id)));
                  const double* dat_ptr = dpdata->getPointer(depth_id);
                  for(int k = ghost[2]; k < klast - ghost[2]; k++) {
                     for(int j = ghost[1]; j < jlast - ghost[1]; j++) {
                        for(int i = ghost[0]; i < ilast - ghost[0]; i++) {
                           if (vector3D(mask, i, j, k) == 1) {
                              double calc = 0.125 * (vectorN3D(dat_ptr, i, j, k) + vectorN3D(dat_ptr, i, j, k + 1) + vectorN3D(dat_ptr, i, j + 1, k) + vectorN3D(dat_ptr, i + 1, j, k) + vectorN3D(dat_ptr, i, j + 1, k + 1) + vectorN3D(dat_ptr, i + 1, j, k + 1) + vectorN3D(dat_ptr, i + 1, j + 1, k) + vectorN3D(dat_ptr, i + 1, j + 1, k + 1));
                              if(std::find(d_calculation.begin(), d_calculation.end(), "INTEGRAL") != d_calculation.end()) {
                                 integral = integral + dx[0]*dx[1]*dx[2] * calc;
                              }
                              if(std::find(d_calculation.begin(), d_calculation.end(), "L2NORM") != d_calculation.end()) {
                                 norm = norm + dx[0]*dx[1]*dx[2] * calc*calc;
                              }
                              if(std::find(d_calculation.begin(), d_calculation.end(), "MIN") != d_calculation.end()) {
                                 min = std::min(min, vectorN3D(dat_ptr, i, j, k));
                              }
                              if(std::find(d_calculation.begin(), d_calculation.end(), "MAX") != d_calculation.end()) {
                                 max = std::max(max, vectorN3D(dat_ptr, i, j, k));
                              }
                              if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                                 absmin = std::min(absmin, fabs(vectorN3D(dat_ptr, i, j, k)));
                              }
                              if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMAX") != d_calculation.end()) {
                                 absmax = std::max(absmax, fabs(vectorN3D(dat_ptr, i, j, k)));
                              }
                           }
                        }
                     }
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "INTEGRAL") != d_calculation.end()) {
                     ipi->d_integral_value[depth_id] = ipi->d_integral_value[depth_id] + integral;
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "L2NORM") != d_calculation.end()) {
                     ipi->d_l2norm_value[depth_id] = ipi->d_l2norm_value[depth_id] + norm;
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "MIN") != d_calculation.end()) {
                     ipi->d_min_value[depth_id] = std::min(min, ipi->d_min_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "MAX") != d_calculation.end()) {
                     ipi->d_max_value[depth_id] = std::max(max, ipi->d_max_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                     ipi->d_absmin_value[depth_id] = std::min(absmin, ipi->d_absmin_value[depth_id]);
                  }
                  if(std::find(d_calculation.begin(), d_calculation.end(), "ABSMIN") != d_calculation.end()) {
                     ipi->d_absmax_value[depth_id] = std::max(absmax, ipi->d_absmax_value[depth_id]);
                  }
               }
            }
         }
      }
   } else {
    TBOX_ERROR("IntegrateDataWriter::writePlotData"
         << "\n    only 2D and 3D is available for integration." << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Private function to clear previous calculations
 *
 *************************************************************************
 */
void IntegrateDataWriter::clearValues() {
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
      for (int i = 0; i < ipi->d_depth; i++) {
         ipi->d_integral_value[i] = 0;
         ipi->d_l2norm_value[i] = 0;
         ipi->d_min_value[i] = std::numeric_limits<double>::max();
         ipi->d_max_value[i] = std::numeric_limits<double>::min();
         ipi->d_absmin_value[i] = std::numeric_limits<double>::max();
         ipi->d_absmax_value[i] = 0;
      }
   }
}

#if !defined(__BGL_FAMILY__) && defined(__xlC__)
/*
 * Suppress XLC warnings
 */
#pragma report(enable, CPPC5334)
#pragma report(enable, CPPC5328)
#endif


#endif
