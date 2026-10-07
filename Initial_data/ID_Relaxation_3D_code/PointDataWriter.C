/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2013 Lawrence Livermore National Security, LLC
 * Description:   Writes data files for visualization by VisIt
 *
 ************************************************************************/

#ifndef included_appu_PointDataWriter_C
#define included_appu_PointDataWriter_C

#include "PointDataWriter.h"
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

#define vector3D(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]

const float PointDataWriter::POINT_DATAWRITER_VERSION_NUMBER = 2.0;
const int PointDataWriter::POINT_NAME_BUFSIZE = 128;
const int PointDataWriter::POINT_UNDEFINED_INDEX = -1;
const int PointDataWriter::POINT_MASTER = 0;

tbox::StartupShutdownManager::Handler
PointDataWriter::s_initialize_handler(PointDataWriter::initializeCallback, 0, 0, PointDataWriter::finalizeCallback, tbox::StartupShutdownManager::priorityTimers);

std::shared_ptr<tbox::Timer> PointDataWriter::t_write_plot_data;

using namespace std;

/*
 *************************************************************************
 *
 * The constructor --- sets default object state.
 *
 *************************************************************************
 */

PointDataWriter::PointDataWriter(
   const std::vector<double> coordinates,
   const std::string& object_name,
   const std::string& dump_directory_name,
   int number_procs_per_file,
   bool is_multiblock):
   d_coordinates(coordinates.begin(), coordinates.end())
{
   TBOX_ASSERT(!object_name.empty());
   TBOX_ASSERT(number_procs_per_file > 0);

   d_object_name = object_name;

   d_time_step_number = POINT_UNDEFINED_INDEX;
   d_top_level_directory_name = dump_directory_name;
}

/*
 *************************************************************************
 *
 * The destructor implicitly deallocates the list of plot data items.
 *
 *************************************************************************
 */

PointDataWriter::~PointDataWriter()
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
PointDataWriter::registerPlotQuantity(
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
         TBOX_ERROR("PointDataWriter::registerPlotQuantity()"
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
PointDataWriter::resetLevelPlotQuantity(
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
   variable_data_type vdt = POINT_DATA_TYPE_BAD;
   variable_centering vc = POINT_CENTERING_BAD;

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));

      if (ffactory) {
         vdt = POINT_FLOAT;
         vc = POINT_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
      if (ffactory) {
         vdt = POINT_FLOAT;
         vc = POINT_NODE;
         found_type = true;
      }
   }

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = POINT_DOUBLE;
         vc = POINT_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = POINT_DOUBLE;
         vc = POINT_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = POINT_INT;
         vc = POINT_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = POINT_INT;
         vc = POINT_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      TBOX_ERROR("PointDataWriter::resetLevelPlotQuantity()"
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
            TBOX_ERROR("PointDataWriter::resetLevelPlotQuantity()"
               << "\n     The supplied patch data id has a different"
               << "\n     type and centering from the one originally"
               << "\n     registered.  hier::Variable name: "
               << variable_name
               << "\n     ***Exiting" << std::endl);
         }
      }
   }

   if (!found_var) {
      TBOX_ERROR("PointDataWriter::resetLevelPlotQuantity()"
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
PointDataWriter::initializePlotItem(
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
      plotitem.d_var_type = POINT_SCALAR;
      plotitem.d_depth = 1;
   } else {
      TBOX_ERROR("PointDataWriter::registerPlotItem"
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
         TBOX_ERROR("PointDataWriter::registerPlotItem"
            << "\n    patch data array index = " << patch_data_index
            << "\n    for variable = " << variable_name
            << "\n    is invalid" << std::endl);
      } else {

         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = POINT_CELL;
               plotitem.d_var_data_type = POINT_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = POINT_CELL;
               plotitem.d_var_data_type = POINT_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = POINT_CELL;
               plotitem.d_var_data_type = POINT_INT;
               var_depth = ifactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = POINT_NODE;
               plotitem.d_var_data_type = POINT_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = POINT_NODE;
               plotitem.d_var_data_type = POINT_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = POINT_NODE;
               plotitem.d_var_data_type = POINT_INT;
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
            TBOX_ERROR("PointDataWriter::registerPlotItem"
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
         plotitem.d_var_centering = POINT_UNKNOWN_CELL;
      } else if (variable_centering == "NODE") {
         plotitem.d_var_centering = POINT_UNKNOWN_NODE;
      } else {
         TBOX_ERROR("PointDataWriter::registerPlotItem"
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
      plotitem.d_var_data_type = POINT_DOUBLE;
   }
   plotitem.d_patch_data_index = patch_data_index;
   plotitem.d_visit_var_name.resize(plotitem.d_depth);
   plotitem.d_value.resize(plotitem.d_depth);

   char temp_buf[POINT_NAME_BUFSIZE];
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
PointDataWriter::writePlotData(
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
            "PointDataWriter: Encountered existing limitation of PointDataWriter\n"
            << "This class cannot write files unless all patch levels have\n"
            << "globally sequentialized nodes.  This can be accomplished\n"
            << "by the sequentialize_patch_indices = TRUE input flag in\n"
            << "GriddingAlgorithm.  This problem can (and should\n"
            << "be fixed soon.");

      }
   }

   t_write_plot_data->start();

   if (time_step_number <= d_time_step_number) {
      TBOX_ERROR("PointDataWriter::writePlotData"
         << "\n    data writer with name " << d_object_name
         << "\n    time step number: " << time_step_number
         << " is <= last time step number: " << d_time_step_number
         << std::endl);
   }
   d_time_step_number = time_step_number;

   if (d_top_level_directory_name.empty()) {
      TBOX_ERROR("PointDataWriter::writePlotData"
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
PointDataWriter::writeFiles(
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
   int nprocs = mpi.getSize();
   tbox::Utilities::recursiveMkdir(d_top_level_directory_name);
   clearValues();
   calculatePoint(hierarchy);
   //Communicate maximum level of value
   int *max_levels = (int *)malloc(sizeof(int) * nprocs);
   mpi.Allgather(&d_at_level, 1, MPI_INT, max_levels, 1, MPI_INT);
   int max_level = -1;
   int proc_index = -1;
   for (int i = 0; i < nprocs; i++) {
      if (max_level < max_levels[i]) {
         max_level = max_levels[i];
         proc_index = i;
      }
   }

   if (my_proc == proc_index) {
      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
         for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
            std::string varname = d_top_level_directory_name + "/" + ipi->d_visit_var_name[depth_id];
            ofstream outputfile;
            char name[1024];
            std::string varname_int = varname + "_POINT";
            strcpy(name, varname_int.c_str());
            // Create file if does not exist, otherwise open at the end of the file
            if (simulation_time == 0) {
               outputfile.open (name, ios::out);
            }
            else {
               outputfile.open (name, ios::out | ios::app);
            }
            outputfile << simulation_time << "\t" << ipi->d_value[depth_id]<< std::endl;
            outputfile.close();
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
PointDataWriter::calculatePoint(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy)
{
   if (hierarchy->getDim().getValue() == 2) {
     for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ln++) {
         std::shared_ptr<hier::PatchLevel> patch_level(hierarchy->getPatchLevel(ln));
         for (hier::PatchLevel::iterator ip(patch_level->begin()); ip != patch_level->end(); ++ip) {
            const std::shared_ptr< hier::Patch >& patch = *ip;
            const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
            const double* dx  = patch_geom->getDx();

            const double* patch_geom_xlo = patch_geom->getXLower();
            const double* patch_geom_xup  = patch_geom->getXUpper();

            if (patch_geom_xlo[0] <= d_coordinates[0] && patch_geom_xup[0] > d_coordinates[0] && patch_geom_xlo[1] <= d_coordinates[1] && patch_geom_xup[1] > d_coordinates[1]) {
               for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
                  for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
                     int patch_data_id = ipi->d_patch_data_index;
                     std::shared_ptr<const pdat::NodeData<double> > dpdata(SAMRAI_SHARED_PTR_CAST<const pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(patch_data_id)));
                     const double* dat_ptr = dpdata->getPointer(depth_id);

                     const hier::Index boxfirst = dpdata->getGhostBox().lower();
                     const hier::Index boxlast  = dpdata->getGhostBox().upper();

                     const hier::IntVector ghost = dpdata->getGhostCellWidth();

                     int ilast = boxlast(0)-boxfirst(0) + 2;
                     int jlast = boxlast(1)-boxfirst(1) + 2;

                     int i = round((d_coordinates[0] - patch_geom_xlo[0])/dx[0]) + ghost[0];
                     int j = round((d_coordinates[1] - patch_geom_xlo[1])/dx[1]) + ghost[1];

                     ipi->d_value[depth_id] = vector2D(dat_ptr, i, j);
                     d_at_level = ln;
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
            const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
            const double* dx  = patch_geom->getDx();

            const double* patch_geom_xlo = patch_geom->getXLower();
            const double* patch_geom_xup  = patch_geom->getXUpper();
            if (patch_geom_xlo[0] <= d_coordinates[0] && patch_geom_xup[0] > d_coordinates[0] && patch_geom_xlo[1] <= d_coordinates[1] && patch_geom_xup[1] > d_coordinates[1] && patch_geom_xlo[2] <= d_coordinates[2] && patch_geom_xup[2] > d_coordinates[2]) {
               for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
                  for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
                     int patch_data_id = ipi->d_patch_data_index;
                     std::shared_ptr<const pdat::NodeData<double> > dpdata(SAMRAI_SHARED_PTR_CAST<const pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(patch_data_id)));
                     const double* dat_ptr = dpdata->getPointer(depth_id);
                     const hier::Index boxfirst = dpdata->getGhostBox().lower();
                     const hier::Index boxlast  = dpdata->getGhostBox().upper();

                     const hier::IntVector ghost = dpdata->getGhostCellWidth();

                     int ilast = boxlast(0)-boxfirst(0) + 2;
                     int jlast = boxlast(1)-boxfirst(1) + 2;
                     int i = round((d_coordinates[0] - patch_geom_xlo[0])/dx[0]) + ghost[0];
                     int j = round((d_coordinates[1] - patch_geom_xlo[1])/dx[1]) + ghost[1];
                     int k = round((d_coordinates[2] - patch_geom_xlo[2])/dx[2]) + ghost[2];
                     ipi->d_value[depth_id] = vector3D(dat_ptr, i, j, k);
                     d_at_level = ln;
                  }
               }
            }
         }
      }
   } else {
    TBOX_ERROR("PointDataWriter::writePlotData"
         << "\n    only 2D and 3D is available for point output." << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Private function to clear previous calculations
 *
 *************************************************************************
 */
void PointDataWriter::clearValues() {
   d_at_level = -1;
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
      for (int i = 0; i < ipi->d_depth; i++) {
         ipi->d_value[i] = 0;
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
