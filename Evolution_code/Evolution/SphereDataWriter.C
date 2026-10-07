#ifndef included_appu_SphereDataWriter_C
#define included_appu_SphereDataWriter_C

#include "SphereDataWriter.h"

#ifdef HAVE_HDF5

#include "SAMRAI/tbox/TimerManager.h"
#include "SAMRAI/hier/BoxLevelConnectorUtils.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/RealBoxConstIterator.h"
#include "SAMRAI/hier/VariableDatabase.h"
#include "SAMRAI/pdat/CellDataFactory.h"
#include "SAMRAI/pdat/NodeDataFactory.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
 #include "SAMRAI/geom/CartesianPatchGeometry.h"


#include <cstring>
#include <ctime>
#include <vector>
#include <math.h>

#define vector(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]
#define vector2D(v, i, j) (v)[i+column*(j)]

const float SphereDataWriter::SPHERE_DATAWRITER_VERSION_NUMBER = 2.0;
const int SphereDataWriter::SPHERE_NAME_BUFSIZE = 128;
const int SphereDataWriter::SPHERE_UNDEFINED_INDEX = -1;
const int SphereDataWriter::SPHERE_MASTER = 0;
const int SphereDataWriter::SPHERE_FILE_CLUSTER_WRITE_BATON = 117;

bool SphereDataWriter::s_summary_file_opened = false;

tbox::StartupShutdownManager::Handler
SphereDataWriter::s_initialize_handler(SphereDataWriter::initializeCallback, 0, 0, SphereDataWriter::finalizeCallback, tbox::StartupShutdownManager::priorityTimers);

std::shared_ptr<tbox::Timer> SphereDataWriter::t_write_plot_data;

/*
 *************************************************************************
 *
 * The constructor --- sets default object state.
 *
 *************************************************************************
 */

SphereDataWriter::SphereDataWriter(
   const std::string& object_name,
   const std::string& dump_directory_name,
    const double radius,
    const std::vector<double> center,
    const std::vector<int> resolution):
   d_dim(3), d_center(center.begin(), center.end()), d_resolution(resolution.begin(), resolution.end())
{
   TBOX_ASSERT(!object_name.empty());

    if (radius <= 0) {
      TBOX_ERROR(
         "SphereDataWriter::SphereDataWriter"
         << "\n          Radius must be grater than 0" << std::endl);
   }

   d_object_name = object_name;

   d_number_working_slaves = SPHERE_UNDEFINED_INDEX;

   d_scaling_ratios.resize(1, hier::IntVector::getOne(d_dim));

   d_number_visit_variables = 0;
   d_number_visit_variables_plus_depth = 0;
   d_number_species = 0;

   d_time_step_number = SPHERE_UNDEFINED_INDEX;
   d_grid_type = SPHERE_CARTESIAN;
   d_top_level_directory_name = dump_directory_name;
   d_summary_filename = "summary.samrai";
   d_number_levels = 1;

   d_radius = radius;
   delta_phi = M_PI / d_resolution[1];
   delta_theta = 2 * M_PI / d_resolution[0];

   d_worker_min_max = 0;
}

/*
 *************************************************************************
 *
 * The destructor implicitly deallocates the list of plot data items.
 *
 *************************************************************************
 */

SphereDataWriter::~SphereDataWriter()
{
   /*
    * De-allocate min/max structs for each variable.
    */
   if (d_worker_min_max != 0)
      delete[] d_worker_min_max;

   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
        ipi != d_plot_items.end(); ipi++) {
      for (int comp = 0; comp < SPHERE_FIXED_DIM; comp++) {
         if (ipi->d_master_min_max[comp] != 0)
            delete[] ipi->d_master_min_max[comp];
      }
   }
}

/*
 *************************************************************************
 *
 * Register plot quantities: scalar, vector or tensor.
 *
 *************************************************************************
 */
void
SphereDataWriter::registerPlotQuantity(
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
         TBOX_ERROR("SphereDataWriter::registerPlotQuantity()"
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

   d_number_visit_variables++;
   d_number_visit_variables_plus_depth += plotitem.d_depth;
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
SphereDataWriter::resetLevelPlotQuantity(
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
   variable_data_type vdt = SPHERE_DATA_TYPE_BAD;
   variable_centering vc = SPHERE_CENTERING_BAD;

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));

      if (ffactory) {
         vdt = SPHERE_FLOAT;
         vc = SPHERE_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
      if (ffactory) {
         vdt = SPHERE_FLOAT;
         vc = SPHERE_NODE;
         found_type = true;
      }
   }

   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = SPHERE_DOUBLE;
         vc = SPHERE_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         vdt = SPHERE_DOUBLE;
         vc = SPHERE_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = SPHERE_INT;
         vc = SPHERE_CELL;
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
      if (ifactory) {
         vdt = SPHERE_INT;
         vc = SPHERE_NODE;
         found_type = true;
      }
   }
   if (!found_type) {
      TBOX_ERROR("SphereDataWriter::resetLevelPlotQuantity()"
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
            TBOX_ERROR("SphereDataWriter::resetLevelPlotQuantity()"
               << "\n     The supplied patch data id has a different"
               << "\n     type and centering from the one originally"
               << "\n     registered.  hier::Variable name: "
               << variable_name
               << "\n     ***Exiting" << std::endl);
         }
         if (level_number >=
             static_cast<int>(ipi->d_level_patch_data_index.size())) {
            ipi->d_level_patch_data_index.resize(level_number + 1);
         }
         ipi->d_level_patch_data_index[level_number] = patch_data_index;
         ipi->d_start_depth_index = start_depth_index;
      }
   }

   if (!found_var) {
      TBOX_ERROR("SphereDataWriter::resetLevelPlotQuantity()"
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
 * Register node coordinates of deformed (moving) grids.
 *
 *************************************************************************
 */

void
SphereDataWriter::registerNodeCoordinates(
   const int patch_data_index,
   const int start_depth_index)
{
   TBOX_ASSERT(patch_data_index >= -1);
   TBOX_ASSERT(start_depth_index >= 0);

   /*
    * Check to make sure "Coords" variable has not already been registered.
    */
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
        ipi != d_plot_items.end(); ipi++) {
      if (ipi->d_var_name == "Coords") {
         TBOX_ERROR("SphereDataWriter::registerNodeCoordinates()"
            << "\n   Coordinates registered more than once." << std::endl);
      }
   }

   /*
    * Set the grid type for the visit data
    */
   d_grid_type = SPHERE_DEFORMED;

   /*
    * Verify the supplied patch data index is a valid NODE-centered
    * float or double and has a depth of at least d_dim
    */
   std::shared_ptr<hier::PatchDataFactory> factory(
      hier::VariableDatabase::getDatabase()->
      getPatchDescriptor()->
      getPatchDataFactory(patch_data_index));

   bool found_type = false;
   int var_depth = SPHERE_UNDEFINED_INDEX;
   if (!found_type) {

      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
      if (ffactory) {
         var_depth = ffactory->getDepth();
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         var_depth = dfactory->getDepth();
         found_type = true;
      }
   }
   if (!found_type) {
      TBOX_ERROR("SphereDataWriter::registerNodeCoordinates"
         << "\n     This variable is NOT a node centered"
         << "\n     float or double type, which is required."
         << "\n     ***Exiting" << std::endl);
   }

   int end_depth = start_depth_index + d_dim.getValue();
   if (var_depth < (end_depth)) {
      TBOX_ERROR("SphereDataWriter::registerNodeCoordinates"
         << "\n     This variable has depth: " << var_depth
         << "\n     It must be a VECTOR type and therefore"
         << "\n     have depth at least d_dim + start_depth_index = "
         << end_depth
         << "\n     ***Exiting" << std::endl);
   }

   /*
    * Create the coords plot item.
    */
   VisItItem plotitem;

   std::string var_name = "Coords";
   std::string var_type = "VECTOR";
   double scale_factor = 1.0;
   std::string var_cent = "NODE";

   initializePlotItem(plotitem,
      var_name,
      var_type,
      patch_data_index,
      start_depth_index,
      scale_factor,
      var_cent);

   plotitem.d_is_deformed_coords = true;

   /*
    * We need to reset the variable name, because it has to be written with
    * a special form to the VisIt readible HDF file.
    */
   char temp_buf[SPHERE_NAME_BUFSIZE];
   for (int i = 0; i < plotitem.d_depth; i++) {
      sprintf(temp_buf, ".%02d", i);
      plotitem.d_visit_var_name[i] = var_name + temp_buf;
   }

   /*
    * In this method, we assume the scale factor is always 1.0.  If a
    * user would like to choose a scale factor, use the
    * "registerSingleNodeCoordinate()" method.
    */
   plotitem.d_coord_scale_factor.resize(d_dim.getValue(), 1.0);

   d_number_visit_variables++;
   d_number_visit_variables_plus_depth += plotitem.d_depth;
   d_plot_items.push_back(plotitem);

}

/*
 *************************************************************************
 *
 * Register node coordinates of deformed (moving) grids.
 *
 *************************************************************************
 */

void
SphereDataWriter::registerSingleNodeCoordinate(
   const int coordinate_number,
   const int patch_data_index,
   const int depth_index,
   const double scale_factor)
{
   TBOX_ASSERT(coordinate_number >= 0 && coordinate_number < d_dim.getValue());
   TBOX_ASSERT(patch_data_index >= -1);
   TBOX_ASSERT(depth_index >= 0);

   /*
    * Set the grid type for the visit data
    */
   d_grid_type = SPHERE_DEFORMED;

   /*
    * Verify the supplied patch data index is a valid NODE-centered
    * float or double
    */
   std::shared_ptr<hier::PatchDataFactory> factory(
      hier::VariableDatabase::getDatabase()->
      getPatchDescriptor()->
      getPatchDataFactory(patch_data_index));

   bool found_type = false;
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
      if (ffactory) {
         found_type = true;
      }
   }
   if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
      if (dfactory) {
         found_type = true;
      }
   }
   if (!found_type) {
      TBOX_ERROR("SphereDataWriter::registerSingleNodeCoordinate"
         << "\n     This variable is NOT a node centered"
         << "\n     float or double type, which is required."
         << "\n     ***Exiting" << std::endl);
   }

   /*
    * Create the coords plot item.  If its the first time this method
    * is called, initialize the "Coords" plot variable.  If it has
    * already been called before (i.e. coordinate_number > 0) then just
    * reset plot variable parameters as necessary.
    */
   if (coordinate_number == 0) {

      /*
       * Check to make sure "Coords" variable has not already been registered.
       */
      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
           ipi != d_plot_items.end(); ipi++) {
         if (ipi->d_var_name == "Coords") {
            TBOX_ERROR("SphereDataWriter::registerSingleNodeCoordinate()"
               << "\n   Coordinate registered more than once."
               << std::endl);
         }
      }

      VisItItem plotitem;

      std::string var_name = "Coords";
      std::string var_type = "SCALAR";
      std::string var_cent = "NODE";

      initializePlotItem(plotitem,
         var_name,
         var_type,
         patch_data_index,
         depth_index,
         scale_factor,
         var_cent);

      plotitem.d_is_deformed_coords = true;

      /*
       * We need to reset the variable name, because it has to be written with
       * a special form to the VisIt readible HDF file.
       */
      char temp_buf[SPHERE_NAME_BUFSIZE];
      for (int i = 0; i < plotitem.d_depth; i++) {
         sprintf(temp_buf, ".%02d", i);
         plotitem.d_visit_var_name[i] = var_name + temp_buf;
      }

      plotitem.d_coord_scale_factor.resize(d_dim.getValue());
      plotitem.d_coord_scale_factor[coordinate_number] = scale_factor;
      d_number_visit_variables++;
      d_number_visit_variables_plus_depth += plotitem.d_depth;

      d_plot_items.push_back(plotitem);

   } else {

      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
           ipi != d_plot_items.end(); ipi++) {

         if (ipi->d_is_deformed_coords) {

            ipi->d_var_type = SPHERE_VECTOR;
            ipi->d_depth = d_dim.getValue();
            ipi->d_visit_var_name.resize(d_dim.getValue());

            std::string var_name = "Coords";
            char temp_buf[SPHERE_NAME_BUFSIZE];
            sprintf(temp_buf, ".%02d", coordinate_number);
            ipi->d_visit_var_name[coordinate_number] = var_name + temp_buf;

            ipi->d_coord_scale_factor[coordinate_number] = scale_factor;
            d_number_visit_variables_plus_depth += 1;
         }
      }
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
SphereDataWriter::initializePlotItem(
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
      plotitem.d_var_type = SPHERE_SCALAR;
      plotitem.d_depth = 1;
   } else if (variable_type == "VECTOR") {
      plotitem.d_var_type = SPHERE_VECTOR;
      plotitem.d_depth = d_dim.getValue();
   } else if (variable_type == "TENSOR") {
      plotitem.d_var_type = SPHERE_TENSOR;
      plotitem.d_depth = d_dim.getValue() * d_dim.getValue();
   } else {
      TBOX_ERROR("SphereDataWriter::registerPlotItem"
         << "\n    variable_type " << variable_type
         << "\n    is unsupported.  You must use SCALAR, VECTOR, or"
         << "\n    TENSOR.  Exiting***" << std::endl);
   }

   /*
    * Check to make sure we have not exceeded max allowed components.
    */
   int num_old_components = d_number_visit_variables_plus_depth;

   int new_num_components = num_old_components + plotitem.d_depth;
   if (new_num_components > SPHERE_MAX_NUMBER_COMPONENTS) {
      TBOX_ERROR("SphereDataWriter::registerPlotItem"
         << "\n     Unable to register this quantity because it"
         << "\n     the maximum number of variables allowed in"
         << "\n     the VisItWriter was reached:"
         << "\n       current num variables:"
         << num_old_components
         << "\n       variable depth: "
         << plotitem.d_depth
         << "\n     MAX_NUMBER_COMPONENTS: "
         << SPHERE_MAX_NUMBER_COMPONENTS
         << "\n     Contact SAMRAI team for assistance." << std::endl);
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
         TBOX_ERROR("SphereDataWriter::registerPlotItem"
            << "\n    patch data array index = " << patch_data_index
            << "\n    for variable = " << variable_name
            << "\n    is invalid" << std::endl);
      } else {

         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = SPHERE_CELL;
               plotitem.d_var_data_type = SPHERE_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = SPHERE_CELL;
               plotitem.d_var_data_type = SPHERE_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::CellDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::CellDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = SPHERE_CELL;
               plotitem.d_var_data_type = SPHERE_INT;
               var_depth = ifactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<float> > ffactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<float>, hier::PatchDataFactory>(
            factory));
            if (ffactory) {
               plotitem.d_var_centering = SPHERE_NODE;
               plotitem.d_var_data_type = SPHERE_FLOAT;
               var_depth = ffactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<double> > dfactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<double>, hier::PatchDataFactory>(
            factory));
            if (dfactory) {
               plotitem.d_var_centering = SPHERE_NODE;
               plotitem.d_var_data_type = SPHERE_DOUBLE;
               var_depth = dfactory->getDepth();
               found_type = true;
            }
         }
         if (!found_type) {
      std::shared_ptr<pdat::NodeDataFactory<int> > ifactory(
         std::dynamic_pointer_cast<pdat::NodeDataFactory<int>, hier::PatchDataFactory>(
            factory));
            if (ifactory) {
               plotitem.d_var_centering = SPHERE_NODE;
               plotitem.d_var_data_type = SPHERE_INT;
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
            TBOX_ERROR("SphereDataWriter::registerPlotItem"
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
         plotitem.d_var_centering = SPHERE_UNKNOWN_CELL;
      } else if (variable_centering == "NODE") {
         plotitem.d_var_centering = SPHERE_UNKNOWN_NODE;
      } else {
         TBOX_ERROR("SphereDataWriter::registerPlotItem"
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
      plotitem.d_var_data_type = SPHERE_DOUBLE;
   }

   /*
    * Set the patch data index.
    */
   plotitem.d_patch_data_index = patch_data_index;
   plotitem.d_level_patch_data_index.resize(d_number_levels, patch_data_index);

   plotitem.d_visit_var_name.resize(plotitem.d_depth);
   char temp_buf[SPHERE_NAME_BUFSIZE];
   for (int i = 0; i < plotitem.d_depth; i++) {
      if (plotitem.d_depth == 1) {
         plotitem.d_visit_var_name[i] = variable_name;
      } else {
         sprintf(temp_buf, ".%02d", i);
         plotitem.d_visit_var_name[i] = variable_name + temp_buf;
      }
   }

   plotitem.d_scale_factor = scale_factor;
   plotitem.d_start_depth_index = start_depth_index;

   /*
    * Initialize min/max information.
    */
   for (int i = 0; i < SPHERE_FIXED_DIM; i++) {
      plotitem.d_master_min_max[i] = 0;
   }

   /*
    * Set derived, coords, material, and species information NULL.
    * If the variable is any of these types, this information
    * should be set by the appropriate registration functions.
    */
   plotitem.d_is_derived = false;
   plotitem.d_derived_writer = 0;
   plotitem.d_is_deformed_coords = false;

}

/*
 *************************************************************************
 *
 * Write plot data from given hierarchy to HDF file
 *
 *************************************************************************
 */

void
SphereDataWriter::writePlotData(
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
            "SphereDataWriter: Encountered existing limitation of SphereDataWriter\n"
            << "This class cannot write files unless all patch levels have\n"
            << "globally sequentialized nodes.  This can be accomplished\n"
            << "by the sequentialize_patch_indices = TRUE input flag in\n"
            << "GriddingAlgorithm.  This problem can (and should\n"
            << "be fixed soon.");

      }
   }

   t_write_plot_data->start();

   if (time_step_number <= d_time_step_number) {
      TBOX_ERROR("SphereDataWriter::writePlotData"
         << "\n    data writer with name " << d_object_name
         << "\n    time step number: " << time_step_number
         << " is <= last time step number: " << d_time_step_number
         << std::endl);
   }
   d_time_step_number = time_step_number;

   d_number_levels = hierarchy->getNumberOfLevels();

   if (d_number_levels > static_cast<int>(d_scaling_ratios.size())) {
      d_scaling_ratios.resize(d_number_levels, hier::IntVector(d_dim));
   }

   for (int ln = 1; ln <= hierarchy->getFinestLevelNumber(); ln++) {
      d_scaling_ratios[ln] =
         hierarchy->getPatchLevel(ln)->getRatioToCoarserLevel();
   }

   if (d_top_level_directory_name.empty()) {
      TBOX_ERROR("SphereDataWriter::writePlotData"
         << "\n    data writer with name " << d_object_name
         << "\n     Dump Directory Name is not set" << std::endl);
   }

   int num_items_to_plot = d_number_visit_variables_plus_depth;

   if (num_items_to_plot == 0) {
      TBOX_ERROR("SphereDataWriter::writePlotData"
         << "\n    No VisIt variables have been registered."
         << std::endl);
   }

   initializePlotVariableMinMaxInfo(hierarchy);

   writeHDFFiles(hierarchy, simulation_time);

   t_write_plot_data->stop();
}

/*
 *************************************************************************
 *
 * Private function to initialize min/max information for the plot
 * components.  This method will allocate space for the d_mm array on
 * the SPHERE_MASTER processsor (which holds min/max info for all plot
 * variables on all patches) and will allocate the d_worker_min_max array
 * on all processors except the SPHERE_MASTER.  This latter array is used
 * to store data to be sent to the master when summary information is
 * written.
 *
 *************************************************************************
 */

void
SphereDataWriter::initializePlotVariableMinMaxInfo(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy)
{
   TBOX_ASSERT(hierarchy);
   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

   /*
    * Compute max number of patches on this processor.
    */
   int number_local_patches = 0;
   int tot_number_of_patches = 0;

   for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ln++) {
      std::shared_ptr<hier::PatchLevel> patch_level(
         hierarchy->getPatchLevel(ln));
      tot_number_of_patches += patch_level->getGlobalNumberOfPatches();
      for (hier::PatchLevel::iterator ip(patch_level->begin());
           ip != patch_level->end(); ++ip) {
         number_local_patches++;
      }
   }

   int max_number_local_patches = number_local_patches;
   if (mpi.getSize() > 1) {
      mpi.AllReduce(&max_number_local_patches, 1, MPI_MAX);
   }

   /*
    * Determine number of worker processors.  NOTE: subract one because we
    * don't want to count processor zero.
    */
   int count_me_in = 0;
   if (number_local_patches > 0 || mpi.getRank() == SPHERE_MASTER)
      count_me_in = 1;
   d_number_working_slaves = count_me_in;
   if (mpi.getSize() > 1) {
      mpi.AllReduce(&d_number_working_slaves, 1, MPI_SUM);
   }
   d_number_working_slaves -= 1;

   if (mpi.getRank() != SPHERE_MASTER) {

      /*
       * Worker processor:  Allocate an array large enough to hold patch
       * min max information.  Pack array by var_item, component number,
       * level, and local patch number.
       */
      int num_items_to_plot = d_number_visit_variables_plus_depth;

      int num_components = max_number_local_patches * num_items_to_plot;
      if (d_worker_min_max != 0) {
         delete[] d_worker_min_max;
      }
      d_worker_min_max = new patchMinMaxStruct[num_components];
      memset((char *)d_worker_min_max, 0,
         num_components * sizeof(patchMinMaxStruct));
      for (int i = 0; i < num_components; i++) {
         d_worker_min_max[i].patch_data_on_disk = false;
         d_worker_min_max[i].min = tbox::MathUtilities<double>::getMax();
         d_worker_min_max[i].max = tbox::MathUtilities<double>::getMin();
         d_worker_min_max[i].material_composition_code =
            appu::VisMaterialsDataStrategy::VISIT_MIXED;
         d_worker_min_max[i].species_composition_code =
            appu::VisMaterialsDataStrategy::VISIT_MIXED;
      }

   } else {   // (mpi.getRank() == SPHERE_MASTER)

      /*
       * Master processor:  allocate array for each plot item to hold
       * min/max information for ALL patches, on all levels.
       */
      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
           ipi != d_plot_items.end(); ipi++) {

         for (int comp = 0; comp < ipi->d_depth; comp++) {

            /*
             * Create space for master min/max struct, if it doesn't
             * already exist.
             */
            if (ipi->d_master_min_max[comp] != 0) {
               delete[] ipi->d_master_min_max[comp];
            }
            patchMinMaxStruct* mm =
               new patchMinMaxStruct[tot_number_of_patches];
            memset((char *)mm, 0,
               tot_number_of_patches * sizeof(patchMinMaxStruct));
            ipi->d_master_min_max[comp] = mm;

            for (int pn = 0; pn < number_local_patches; pn++) {
               ipi->d_master_min_max[comp][pn].patch_data_on_disk = false;
               ipi->d_master_min_max[comp][pn].min =
                  tbox::MathUtilities<double>::getMax();
               ipi->d_master_min_max[comp][pn].max =
                  tbox::MathUtilities<double>::getMin();
               ipi->d_master_min_max[comp][pn].material_composition_code =
                  appu::VisMaterialsDataStrategy::VISIT_MIXED;
               ipi->d_master_min_max[comp][pn].species_composition_code =
                  appu::VisMaterialsDataStrategy::VISIT_MIXED;
            }
         }
      }
   } // proc == SPHERE_MASTER

}

/*
 *************************************************************************
 *
 * Private function to coordinate writing HDF plot files.
 *
 *************************************************************************
 */

void
SphereDataWriter::writeHDFFiles(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   double simulation_time)
{

   TBOX_ASSERT(hierarchy);

// Disable Intel warning about conversions
#ifdef __INTEL_COMPILER
#pragma warning (disable:810)
#endif

   char temp_buf[SPHERE_NAME_BUFSIZE];
   std::string dump_dirname;
   tbox::Database* visit_HDFFilePointer;

   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
   int my_proc = mpi.getRank();

   sprintf(temp_buf, "%05d", d_time_step_number);
   d_current_dump_directory_name = "visit_dump.";
   d_current_dump_directory_name += temp_buf;
   if (!d_top_level_directory_name.empty() &&
       d_top_level_directory_name[d_top_level_directory_name.length()-1] == '/') {
      dump_dirname = d_top_level_directory_name;
   }
   else {
      dump_dirname = d_top_level_directory_name + "/";
   }
   dump_dirname = dump_dirname + d_current_dump_directory_name;
   tbox::Utilities::recursiveMkdir(dump_dirname);

  std::shared_ptr<tbox::Database> processor_HDFGroup;
  if (my_proc == SPHERE_MASTER) {
        // cluster_leader guaranteed to enter this section before anyone else
        sprintf(temp_buf, "/processor_cluster.%05d.samrai", my_proc);
        std::string database_name(temp_buf);
        std::string visit_HDFFilename = dump_dirname + database_name;
        visit_HDFFilePointer = new tbox::HDFDatabase(database_name);
        // creates the HDF file:
        //      dirname/visit_dump.000n/processor_cluster.0000.samrai
        //      where n is timestep #
        visit_HDFFilePointer->create(visit_HDFFilename);

        // create group for this proc
        sprintf(temp_buf, "processor.%05d", my_proc);
        processor_HDFGroup = visit_HDFFilePointer->putDatabase(std::string(temp_buf));
  }

   writeVisItVariablesToHDFFile(processor_HDFGroup,
       hierarchy);
   if (my_proc == SPHERE_MASTER) {
        visit_HDFFilePointer->close(); // invokes H5FClose
        delete visit_HDFFilePointer; // deletes tbox::HDFDatabase object
   }
   tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();

   writeSummaryToHDFFile(dump_dirname,
      hierarchy,
      0,
      hierarchy->getFinestLevelNumber(),
      simulation_time);
}

/*
 *************************************************************************
 *
 * Private function to find global patch number, given level number and
 * local patch number. This is needed because SAMRAI maintains the patch
 * number on each level, but VisIt needs a unique number for each patch
 * written.
 *
 *************************************************************************
 */

int
SphereDataWriter::getGlobalPatchNumber(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const int level_number,
   const int patch_number)
{
   TBOX_ASSERT(hierarchy);
   TBOX_ASSERT(level_number >= 0);
   TBOX_ASSERT(patch_number >= 0);

   int global_patch_id = 0;

   for (int i = 0; i < level_number; i++) {
      global_patch_id +=
         hierarchy->getPatchLevel(i)->getGlobalNumberOfPatches();
   }

   global_patch_id += patch_number;
   return global_patch_id;
}

/*
 *************************************************************************
 *
 * Private function to write variables data to an HDF File.
 *
 *************************************************************************
 */

void
SphereDataWriter::writeVisItVariablesToHDFFile(
   const std::shared_ptr<tbox::Database>& processor_HDFGroup,
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy)
{
   TBOX_ASSERT(hierarchy);
   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
   /*
    * Reset the var_id_ctr - this is used to record min/max summary
    * information for every plotted variable on the patch.  It is incremented
    * for each component (i.e. depth), of each variable, of each patch.
    */
   d_var_id_ctr = 0;

   char temp_buf[SPHERE_NAME_BUFSIZE];
   std::shared_ptr<tbox::Database> level_HDFGroup, patch_HDFGroup;
   
   const std::shared_ptr<geom::CartesianGridGeometry> ggeom(SAMRAI_SHARED_PTR_CAST<geom::CartesianGridGeometry, hier::BaseGridGeometry>(hierarchy->getGridGeometry()));
   const double* dx  = ggeom->getDx();
   const double* gXLower  = ggeom->getXLower();
   /*
    * create new HDFGroup for this level
    */
   if (mpi.getRank() == SPHERE_MASTER) {
     sprintf(temp_buf, "level.%05d", 0);
     level_HDFGroup = processor_HDFGroup->putDatabase(std::string(temp_buf));

     sprintf(temp_buf, "patch.%05d", 0);
     patch_HDFGroup = level_HDFGroup->putDatabase(std::string(temp_buf));
   }
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
     for (int depth_id = 0; depth_id < ipi->d_depth; depth_id++) {
      int buf_size = (d_resolution[0] + 1) * (d_resolution[1] + 1);
       float* fbuffer = new float[buf_size]; // copy to float for writing
       double* dbuffer = new double[buf_size]; // copy to float for writing
       int* minLevel = new int[buf_size]; // minimum level fror point
       int* minLevelLocal = new int[buf_size]; // minimum level fror point
       int column = d_resolution[0] + 1;

       for (int j = 0; j <= d_resolution[1]; j++) {
          double phi = j * delta_phi;
          for (int i = 0; i <= d_resolution[0]; i++) {
              double theta = i * delta_theta;
              double x = d_center[0] + d_radius * cos(theta) * sin(phi);
              double y = d_center[1] + d_radius * sin(theta) * sin(phi);
              double z = d_center[2] + d_radius * cos(phi);

              // Index of lower left point of interpolation cell:
              double ii = ((x - gXLower[0]) / dx[0]);
              int ibase = floor(ii);
              double jj = ((y - gXLower[1]) / dx[1]);
              int jbase = floor(jj);
              double kk = ((z - gXLower[2]) / dx[2]);
              int kbase = floor(kk);

              vector2D(dbuffer, i, j) = calculateSphericalPoint(hierarchy, ibase, jbase, kbase, x, y, z, depth_id, *ipi, minLevel, i, j, column);
              vector2D(minLevelLocal,i,j) = vector2D(minLevel,i,j);
          }
       }
       tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();

        if (mpi.getSize() > 1) {
          mpi.AllReduce( minLevel, buf_size, MPI_MAX);
        }

       tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();

       for (int j = 0; j <= d_resolution[1]; j++) {
         for (int i = 0; i <= d_resolution[0]; i++) {
            if (vector2D(minLevelLocal,i,j) < vector2D(minLevel,i,j)) {
              vector2D(dbuffer, i, j) = tbox::MathUtilities<double>::getMax();
            }
         }
       }

      tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();      

        //SINCRONIZACION con reducción
        if (mpi.getSize() > 1) {
          mpi.AllReduce( dbuffer, buf_size, MPI_MIN);
        }
        if (mpi.getRank() == SPHERE_MASTER) {
          //Min/Max Information
          double dmax = -tbox::MathUtilities<double>::getMax();
          double dmin = tbox::MathUtilities<double>::getMax();

          /*
           * Scale data (while still double)
           */
          const double scale = ipi->d_scale_factor;
          if (scale != 1.0) {
             for (int i = 0; i < buf_size; i++) {
                dbuffer[i] *= scale;
             }
          }

          /*
           * Determine min/max.
           */
          for (int i = 0; i < buf_size; i++) {
             if (dbuffer[i] > dmax) {
                dmax = dbuffer[i];
             }
             if (dbuffer[i] < dmin) {
                dmin = dbuffer[i];
             }
          }

          checkFloatMinMax(ipi->d_visit_var_name[depth_id],
               dmin,
               dmax);

          /*
           * Convert buffer from double to float
           */
          for (int i = 0; i < buf_size; i++) {
             fbuffer[i] = static_cast<float>(dbuffer[i]);
          }

          /*
           * Write to disk
           */
            std::string vname = ipi->d_visit_var_name[depth_id];
            patch_HDFGroup->putFloatArray(vname,
               fbuffer,
               buf_size);
         /*
          * Write min/max summary info
          */
          ipi->d_master_min_max[depth_id][0].patch_data_on_disk = true;
          ipi->d_master_min_max[depth_id][0].min = dmin;
          ipi->d_master_min_max[depth_id][0].max = dmax;
       }
       delete[] fbuffer;
       delete[] dbuffer;
       delete[] minLevel;
       delete[] minLevelLocal;
     }
   }    

   /*
    * Clean up from packing operations.  If this is not done there is a dangling smart
    * pointer reference to HDF5 groups and the file may not be written/closed.
    */
   for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
        ipi != d_plot_items.end(); ipi++) {
      ipi->d_species_HDFGroup.reset();
      ipi->d_extents_species_HDFGroup.reset();
   }
}

/*
 *************************************************************************
 *
 * Calculates the value of the spherical projection
 * from a 3D point.
 *
 *************************************************************************
 */

double
SphereDataWriter::calculateSphericalPoint(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const int i,
   const int j,
   const int k,
   const double x,
   const double y,
   const double z,
   const int depth_id,
   VisItItem visitItem,
   int* minLevel,
   int a,
   int b,
   int column)
{
   TBOX_ASSERT(hierarchy);

   const std::shared_ptr<geom::CartesianGridGeometry> ggeom(SAMRAI_SHARED_PTR_CAST<geom::CartesianGridGeometry, hier::BaseGridGeometry>(hierarchy->getGridGeometry()));
   const double* gXLower  = ggeom->getXLower();

   //Find the finest level that have the current position
   for (int ln=hierarchy->getFinestLevelNumber(); ln>=0; --ln ) {
	  std::shared_ptr<hier::PatchLevel > level(hierarchy->getPatchLevel(ln));
	  for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
	      const std::shared_ptr<hier::Patch > patch = *p_it;
	      const hier::Index lowers = patch->getBox().lower();
	      const hier::Index uppers = patch->getBox().upper();
	      const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
	      const double* dx  = patch_geom->getDx();

	      // Index of lower left point of interpolation cell:
	      double ii = ((x - gXLower[0]) / dx[0]);
	      int ibase = floor(ii);
	      double jj = ((y - gXLower[1]) / dx[1]);
	      int jbase = floor(jj);
	      double kk = ((z - gXLower[2]) / dx[2]);
	      int kbase = floor(kk);

	      //Check if current patch contains the position
	      if (ibase >= lowers[0] && ibase <= uppers[0] + 1 && jbase >= lowers[1] && jbase <= uppers[1] + 1 && kbase >= lowers[2] && kbase <= uppers[2] + 1) {

		  int patch_data_id = SPHERE_UNDEFINED_INDEX;
		  /*
		   * Check if patch data id has been reset on the level.  If
		   * not, just use the original registered data id.
		   */
		  patch_data_id = visitItem.d_patch_data_index;
		  if (static_cast<int>(visitItem.d_level_patch_data_index.size()) > ln) {
		     patch_data_id = visitItem.d_level_patch_data_index[ln];
		  }

		  bool data_exists_on_patch = patch->checkAllocated(patch_data_id);

		  if (data_exists_on_patch) {
		     int new_depth_id = visitItem.d_start_depth_index + depth_id;
		     std::shared_ptr<hier::PatchData> pdata = patch->getPatchData(patch_data_id);
		     const hier:: IntVector ghosts = pdata->getGhostCellWidth();
		     variable_data_type data_type = visitItem.d_var_data_type;
		      // Fractional distance point is from lower left point:
		      double xfrac, yfrac, zfrac;
		      xfrac = ( x - ((ibase*dx[0]) + gXLower[0])) / dx[0];
		      yfrac = ( y - ((jbase*dx[1]) + gXLower[1])) / dx[1];
		      zfrac = ( z - ((kbase*dx[2]) + gXLower[2])) / dx[2];

		      vector2D(minLevel,a,b) = ln;
		      return interpolate(pdata, data_type, new_depth_id, ibase, jbase, kbase, xfrac, yfrac, zfrac, lowers, ghosts);
		  }
	      }
	  }
   }
   //If the position is not found return maximum double for a MPI_MIN reduction, where the proper value will be selected
   vector2D(minLevel,a,b) = -1;
   return tbox::MathUtilities<double>::getMax();
}

/*
 *************************************************************************
 * Interpolates the spherical point from the indexed cell.
 *************************************************************************
 */

double
SphereDataWriter::interpolate(
   const std::shared_ptr<hier::PatchData> pdata,
   const variable_data_type data_type,
   const int depth_id,
   const int i,
   const int j,
   const int k,
   const double xfrac,
   const double yfrac,
   const double zfrac,
   const hier::Index lowers,
   const hier:: IntVector ghost)
{
    hier::Index databox_lower = pdata->getGhostBox().lower();
    hier::Index databox_upper = pdata->getGhostBox().upper();
    int ilast = databox_upper(0)-databox_lower(0) + 2;
    int jlast = databox_upper(1)-databox_lower(1) + 2;




    switch (data_type) {
          case SPHERE_FLOAT: {
             const std::shared_ptr<pdat::NodeData<float> > fpdata(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<float>, hier::PatchData>(pdata));
             TBOX_ASSERT(fpdata);
             float* dat_ptr = fpdata->getPointer(depth_id);
             return (1 - xfrac) * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac  * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) * (1-yfrac) *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2]);
          }

          case SPHERE_DOUBLE: {
             const std::shared_ptr<pdat::NodeData<double> > dpdata(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(pdata));
             TBOX_ASSERT(dpdata);
             double* dat_ptr = dpdata->getPointer(depth_id);
             return (1 - xfrac) * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac  * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) * (1-yfrac) *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2]);
          }

          case SPHERE_INT: {
             const std::shared_ptr<pdat::NodeData<int> > ipdata(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<int>, hier::PatchData>(pdata));
             TBOX_ASSERT(ipdata);
             int* dat_ptr = ipdata->getPointer(depth_id);
             return (1 - xfrac) * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac  * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    +        xfrac  * (1-yfrac) * (1-zfrac) * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) *      yfrac  * (1-zfrac) * vector(dat_ptr, i - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k - lowers[2] + ghost[2])
                    + (1 - xfrac) * (1-yfrac) *      zfrac  * vector(dat_ptr, i - lowers[0] + ghost[0], j - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2])
                    +        xfrac  *      yfrac *      zfrac  * vector(dat_ptr, i + 1 - lowers[0] + ghost[0], j + 1 - lowers[1] + ghost[1], k + 1 - lowers[2] + ghost[2]);
          }

          default: {
             TBOX_ERROR(
                d_object_name << ":calculateSphericalPoint()"
                              << "\n  Unknown type.  ***Exiting."
                              << std::endl);
          }
    }
}


/*
 *************************************************************************
 *
 * Private function to check float min/max values.
 *
 *************************************************************************
 */

void
SphereDataWriter::checkFloatMinMax(
   const std::string& var_name,
   const double dmin,
   const double dmax)
{

   double fmin = -(tbox::MathUtilities<double>::getMax());
   double fmax = tbox::MathUtilities<double>::getMax();

   if (dmin < fmin) {
      TBOX_ERROR("SphereDataWriter:"
         << "\n    hier::Patch data "
         << var_name
         << " is less than FLT_MIN "
         << "  value: " << dmin
         << "\n    It cannot be read by VisIt."
         << "\n    Make sure data is properly initialized or"
         << "\n    use scale factor to increase its size.");
   }
   if (dmax > fmax) {
      TBOX_ERROR("SphereDataWriter:"
         << "\n    hier::Patch data "
         << var_name
         << " is greater than FLT_MAX "
         << "  value: " << dmax
         << "\n    It cannot be interpreted by VisIt."
         << "\n    Make sure data is properly initialized or"
         << "\n    use scale factor to decrease its size.");
   }
}

/*
 *************************************************************************
 *
 * Private function to write one summary HDF file covering data from all
 * processors for use by VisIt.
 *
 *************************************************************************
 */

void
SphereDataWriter::writeSummaryToHDFFile(
   std::string dump_dirname,
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   int coarsest_plot_level,
   int finest_plot_level,
   double simulation_time)
{
   TBOX_ASSERT(hierarchy);
   TBOX_ASSERT(coarsest_plot_level >= 0);
   TBOX_ASSERT(finest_plot_level >= 0);

   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
   int i, ln;

   /*
    * The "SPHERE_MASTER" writes a set of summary information to
    * the summary file that describes data contained in the visit
    * files written by each MPI process.
    *
    * Although the SPHERE_MASTER processor needs the global mesh
    * data, global communication is required, so access the
    * global data to make sure it is built.
    */
   for (ln = 0; ln < hierarchy->getNumberOfLevels(); ++ln) {
      hierarchy->getPatchLevel(ln)->getBoxes();
   }
   int my_proc = mpi.getRank();

   if (my_proc == SPHERE_MASTER) {
      char temp_buf[SPHERE_NAME_BUFSIZE];
      //sprintf(temp_buf, "/summary.samrai");
      //string summary_HDFFilename = dump_dirname + temp_buf;
      std::string summary_HDFFilename = dump_dirname + "/" + d_summary_filename;
      std::shared_ptr<tbox::Database> summary_HDFFilePointer(
         std::make_shared<tbox::HDFDatabase>("root"));
      summary_HDFFilePointer->create(summary_HDFFilename);

      /*
       * Create BASIC information HDF Group and provide it the following
       * information:
       *   - VisItWriter version number
       *   - grid_type (CARTESIAN or DEFORMED)
       *   - simulation time
       *   - time step number
       *   - number of processors
       *   - number of file clusters
       *   - DIM
       *   - number of levels
       *   - number of patches at each level (array - int[nlevels])
       *   - total number of patches (on all levels)
       *   - ratio to coarser level (array - int[nlevels][ndim])
       */

      sprintf(temp_buf, "BASIC_INFO");
      std::shared_ptr<tbox::Database> basic_HDFGroup(
         summary_HDFFilePointer->putDatabase(std::string(temp_buf)));

      std::shared_ptr<tbox::HDFDatabase> hdf_database(
         SAMRAI_SHARED_PTR_CAST<tbox::HDFDatabase, tbox::Database>(basic_HDFGroup));
      TBOX_ASSERT(hdf_database);
      hid_t basic_group_id = hdf_database->getGroupId();

      std::string key_string = "VDR_version_number";
      basic_HDFGroup->putFloat(key_string, SPHERE_DATAWRITER_VERSION_NUMBER);

      key_string = "grid_type";
      std::string data_string;
      if (d_grid_type == SPHERE_CARTESIAN) {
         data_string = "CARTESIAN";
      } else if (d_grid_type == SPHERE_DEFORMED) {
         data_string = "DEFORMED";
      } else {
         TBOX_ERROR("SphereDataWriter::writeSummaryToHDFFile"
            << "\n    data writer with name " << d_object_name
            << "\n    Illegal grid type: " << d_grid_type << std::endl);
      }
      basic_HDFGroup->putString(key_string, data_string);

      key_string = "time";
      basic_HDFGroup->putDouble(key_string, simulation_time);

      key_string = "time_step_number";
      basic_HDFGroup->putInteger(key_string, d_time_step_number);

      key_string = "number_processors";
      basic_HDFGroup->putInteger(key_string, 1);

      key_string = "number_file_clusters";
      basic_HDFGroup->putInteger(key_string, 1);

      key_string = "number_dimensions_of_problem";
      basic_HDFGroup->putInteger(key_string, d_dim.getValue() - 1);

      key_string = "number_levels";

      basic_HDFGroup->putInteger(key_string, 1);

      key_string = "number_patches_at_level";
      std::vector<int> num_patches_per_level(1);
      num_patches_per_level[0] = 1;
      basic_HDFGroup->putIntegerVector(key_string, num_patches_per_level);

      key_string = "number_global_patches";
      basic_HDFGroup->putInteger(key_string, 1);

      /*
       * When writing VisIt data, it expects to see 3D data for
       * xlo, dx, ratios_to_coarser, and number of ghosts.  The
       * SPHERE_FIXED_DIM is set to 3, and the third element
       * is zero if we have 2D data.
       */
      key_string = "ratios_to_coarser_levels";
      int idx = 0;
      int* rtcl = new int[SPHERE_FIXED_DIM];
      for (i = 0; i < SPHERE_FIXED_DIM; i++) rtcl[i] = 0;
      for (i = 0; i < SPHERE_FIXED_DIM - 1; i++) rtcl[i] = SPHERE_UNDEFINED_INDEX;
      HDFputIntegerArray2D(key_string,
         rtcl,
         1,
         SPHERE_FIXED_DIM,
         basic_group_id);
      delete[] rtcl;

      /*
       * Write to BASIC HDF group information about the
       * VisIt plot variables:
       *   - number of visit variables
       *
       *   for each variable {
       *      - name
       *      - centering (1 = CELL, 0 = NODE)
       *      - scale factor
       *      - depth
       *      - ghosts
       *   }
       */
      key_string = "number_visit_variables";
      basic_HDFGroup->putInteger(key_string, d_number_visit_variables);

      std::vector<std::string> var_names(d_number_visit_variables);
      std::vector<int> var_centering(d_number_visit_variables);
      std::vector<double> var_scale_factors(d_number_visit_variables);
      std::vector<int> var_depths(d_number_visit_variables);

      // SGS propose adding array indicating clean/mixed
      std::vector<int> var_material_state_variable(d_number_visit_variables);

      int* var_ghosts = new int[d_number_visit_variables * SPHERE_FIXED_DIM];
      for (i = 0; i < d_number_visit_variables * SPHERE_FIXED_DIM; i++) {
         var_ghosts[i] = 0;
      }

      i = 0;
      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin()); ipi != d_plot_items.end(); ipi++) {
          var_names[i] = ipi->d_var_name;
          if ((ipi->d_var_centering == SPHERE_CELL) ||
              (ipi->d_var_centering == SPHERE_UNKNOWN_CELL)) {
             var_centering[i] = 1;
          } else {
             var_centering[i] = 0;
          }
          if (ipi->d_is_deformed_coords) {
             var_scale_factors[i] = 0.0;
          } else {
             var_scale_factors[i] = ipi->d_scale_factor;
          }
          var_depths[i] = ipi->d_depth;
          for (int dim = 0; dim < d_dim.getValue(); dim++) {
             var_ghosts[i * SPHERE_FIXED_DIM + dim] =
                0;
          }
           var_material_state_variable[i] = 0;
          i++;
      }

      key_string = "var_names";
      basic_HDFGroup->putStringVector(key_string, var_names);

      key_string = "var_cell_centered";
      basic_HDFGroup->putIntegerVector(key_string, var_centering);

      key_string = "scaling";
      basic_HDFGroup->putDoubleVector(key_string, var_scale_factors);

      // VCHANGE VisIt needs to read this array so it will
      //         know which variables have mixed material data.
      // SGS
      key_string = "material_state_variable";
      basic_HDFGroup->putIntegerVector(key_string,
         var_material_state_variable);

      if (d_grid_type == SPHERE_DEFORMED) {
         std::vector<double> coord_scaling(SPHERE_FIXED_DIM);
         for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
              ipi != d_plot_items.end(); ipi++) {
            if (ipi->d_is_deformed_coords) {
               for (i = 0; i < SPHERE_FIXED_DIM; i++) {
                  coord_scaling[i] = 0.0;
                  if (i < d_dim.getValue()) {
                     coord_scaling[i] = ipi->d_coord_scale_factor[i];
                  }
               }
            }
         }
         key_string = "deformed_coordinate_scaling";
         basic_HDFGroup->putDoubleVector(key_string, coord_scaling);
      }

      key_string = "var_number_components";
      basic_HDFGroup->putIntegerVector(key_string, var_depths);

      key_string = "var_number_ghosts";
      HDFputIntegerArray2D(key_string,
         var_ghosts,
         d_number_visit_variables,
         SPHERE_FIXED_DIM,
         basic_group_id);
      delete[] var_ghosts;

      /*
       * Write time and data info to BASIC HDF group
       */
      key_string = "time_of_dump";

      const int MAXLEN = 256;
      char s[MAXLEN];
      time_t t = time(0);
      strftime(s, MAXLEN, "%a %b %d %H:%M:%S %Z %Y", localtime(&t));
      basic_HDFGroup->putString(key_string, std::string(s));

   
      /*
       * When writing VisIt data, it expects to see 3D data for
       * xlo, dx, ratios_to_coarser, and number of ghosts.  The
       * SPHERE_FIXED_DIM is set to 3, and the third element
       * is zero if we have 2D data.
       */
      double geom_lo[SPHERE_FIXED_DIM] = { 0., 0., 0. };

      double dx_curr_lev[SAMRAI::MAX_DIM_VAL];
      double patch_xlo, patch_xhi;

      for (i = 0; i < d_dim.getValue(); i++) {
         dx_curr_lev[i] = 0.0;
      }

      /*
       * Add mesh dx information to BASIC group
       */

      double* dx = new double[SPHERE_FIXED_DIM];
      double* xlo = new double[SPHERE_FIXED_DIM];
      for (i = 0; i < SPHERE_FIXED_DIM; i++) {
         dx[i] = 0.0;
         xlo[i] = 0.0;
      }
      dx[0] = delta_theta;
      dx[1] = delta_phi;
      
      key_string = "dx";
      HDFputDoubleArray2D(key_string,
         dx,
         1,
         SPHERE_FIXED_DIM,
         basic_group_id);
      delete[] dx;

      /*
       * Add mesh xlo information to BASIC group
       */
      key_string = "XLO";
      basic_HDFGroup->putDoubleArray(key_string,
         xlo,
         SPHERE_FIXED_DIM);
      delete[] xlo;


     key_string = "child_array_length";
     basic_HDFGroup->putInteger(key_string, 0);
     key_string = "parent_array_length";
     basic_HDFGroup->putInteger(key_string, 0);

      /*
       * Write processor mapping information and domain extents
       * for each patch.
       */

      sprintf(temp_buf, "extents");
      std::shared_ptr<tbox::Database> extents_HDFGroup(
         summary_HDFFilePointer->putDatabase(std::string(temp_buf)));
      hdf_database =
        SAMRAI_SHARED_PTR_CAST<tbox::HDFDatabase, tbox::Database>(extents_HDFGroup);
      TBOX_ASSERT(hdf_database);
      hid_t extents_group_id = hdf_database->getGroupId();

      /*
       * Create "patch_map" sub-database of extents group.
       */
      patchMapStruct* pms = new patchMapStruct[1];

      pms[0].processor_number = 0;
      pms[0].file_cluster_number = 0;
      pms[0].level_number = 0;
      pms[0].patch_number = 0;

      key_string = "patch_map";
      HDFputPatchMapStructArray(key_string,
         pms,
         1,
         extents_group_id);

      delete[] pms;

      /*
       * Printf-style string on how to name patches:
       * %T --- local patch number
       * %L --- level number
       * %B --- block number
       * %P --- processor number
       * %F --- file number
       */
      key_string = "patch_names_printf";
      std::string patch_names_printf = "f%F R%P B%B L%02L P%05T";
      basic_HDFGroup->putString(key_string, patch_names_printf);

      /*
       * Create "patch_extents" sub-database of extents group.
       */
      patchExtentsStruct* pes = new patchExtentsStruct[1];

       for (i = 0; i < SPHERE_FIXED_DIM; i++) {
          pes[0].lower[i] = 0;
          pes[0].upper[i] = 0;
          pes[0].xlo[i] = 0.;
          pes[0].xhi[i] = 0.;
       }
       pes[0].upper[0] = d_resolution[0] - 1;
       pes[0].upper[1] = d_resolution[1] - 1;
       pes[0].xhi[0] = 2 * M_PI;
       pes[0].xhi[1] = M_PI;

      /*
       * Write patch min/max for each variable.
       */
      std::shared_ptr<tbox::Database> extents_materials_HDFGroup;
      for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
           ipi != d_plot_items.end(); ipi++) {
         for (int comp = 0; comp < ipi->d_depth; comp++) {

            /*
             * Regular (i.e. not materials or species) variables
             */

               key_string = ipi->d_visit_var_name[comp] + "-Extents";
               HDFputPatchMinMaxStructArray(
                  key_string,
                  ipi->d_master_min_max[comp],
                  1,
                  extents_group_id);

         } // loop over components
      } // loop over variables

      delete[] d_worker_min_max;

      key_string = "patch_extents";
      HDFputPatchExtentsStructArray(key_string,
         pes,
         1,
         extents_group_id);

      delete[] pes;

      summary_HDFFilePointer->close();

   } // if SPHERE_MASTER

   tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();

   if (my_proc == SPHERE_MASTER) {

      /*
       * Add this dump entry to dumps.visit file
       */
      if (d_time_step_number == 0) s_summary_file_opened = false;
      std::string path;
      if (!d_top_level_directory_name.empty() &&
          d_top_level_directory_name[d_top_level_directory_name.length()-1] == '/') {
         path = d_top_level_directory_name + "dumps.visit";
      }
      else {
         path = d_top_level_directory_name + "/dumps.visit";
      }
      std::string file = d_current_dump_directory_name + "/"
         + d_summary_filename;

      /*
       * If summary file has not yet been opened, open and write file.
       * If it has been opened, append this to file.
       */
      if (!s_summary_file_opened) {
         s_summary_file_opened = true;
         std::ofstream sfile(path.c_str(), std::ios::out);
         sfile << file << "\n";
         sfile.close();
      } else {
         std::ofstream sfile(path.c_str(), std::ios::app);
         sfile << file << "\n";
         sfile.close();
      }
   }

   tbox::SAMRAI_MPI::getSAMRAIWorld().Barrier();
}

/*
 *************************************************************************
 *
 * Private function to store min/max information on each patch for each
 * variable.  The "master" processor allocates an array that will hold
 * the global data.  The "worker" processors send this information to
 * the master which in turn unpacks and stores the data.
 *
 *************************************************************************
 */
void
SphereDataWriter::exchangeMinMaxPatchInformation(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const int coarsest_plot_level,
   const int finest_plot_level)
{
   TBOX_ASSERT(hierarchy);
   TBOX_ASSERT(coarsest_plot_level >= 0);
   TBOX_ASSERT(finest_plot_level >= 0);

   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

   /*
    * Compute max number of patches on any processor, and the total number of
    * patches in the problem.
    */
   int ln, pn, comp, item_ctr;
   int number_local_patches = 0;
   int tot_number_of_patches = 0;

   for (ln = coarsest_plot_level; ln <= finest_plot_level; ln++) {
      std::shared_ptr<hier::PatchLevel> patch_level(
         hierarchy->getPatchLevel(ln));
      tot_number_of_patches += patch_level->getGlobalNumberOfPatches();
      for (hier::PatchLevel::iterator ip(patch_level->begin());
           ip != patch_level->end(); ++ip) {
         number_local_patches++;
      }
   }

   int max_number_local_patches = number_local_patches;
   if (mpi.getSize() > 1) {
      mpi.AllReduce(&max_number_local_patches, 1, MPI_MAX);
   }

   int num_items_to_plot = d_number_visit_variables_plus_depth;

   int message_size = max_number_local_patches
      * static_cast<int>(sizeof(patchMinMaxStruct)) * num_items_to_plot;

   if (mpi.getRank() != SPHERE_MASTER) {

      /*
       * Worker processor:  send contents of d_worker_min_max array that
       * was setup in "initializePlotVariableMinMaxInfo()" to the
       * master processor.
       */
      if (number_local_patches > 0) {
         if (mpi.getSize() > 1) {
            mpi.Send(d_worker_min_max,
               message_size,
               MPI_BYTE,
               SPHERE_MASTER,
               0);
         }
      }

   } else { // (mpi.getRank() == SPHERE_MASTER)

      /*
       * Master processor:  Receive the min/max information sent by the
       * worker processors (above).  Unpack into the local d_mm array
       * for each plot variable.
       */

      // recv buffer large enough to receive info from any processor.
      patchMinMaxStruct* buf = 0;
      if (d_number_working_slaves > 0) {
         buf = new patchMinMaxStruct[max_number_local_patches
                                     * num_items_to_plot];
         memset((char *)buf, 0,
            max_number_local_patches * num_items_to_plot
            * sizeof(patchMinMaxStruct));
      }

      /*
       * Receive information sent by "sending_proc".
       */
      int number_msgs_recvd = 0;
      while (number_msgs_recvd < d_number_working_slaves) {
         int sending_proc = -1;
         if (mpi.getSize() > 1) {
            tbox::SAMRAI_MPI::Status status;
            mpi.Recv(buf,
               message_size,
               MPI_BYTE,
               MPI_ANY_SOURCE,
               MPI_ANY_TAG,
               &status);
            sending_proc = status.MPI_SOURCE;
         }
         number_msgs_recvd++;

         /*
          * Unpack the information from buf and fill d_worker_min_max array.
          */
         item_ctr = 0;

         for (ln = coarsest_plot_level; ln <= finest_plot_level; ln++) {
            const std::vector<int>& proc_mapping =
               hierarchy->getPatchLevel(ln)->getProcessorMapping().getProcessorMapping();

            int npatches_on_level = static_cast<int>(proc_mapping.size());
            for (pn = 0; pn < npatches_on_level; pn++) {
               if (proc_mapping[pn] == sending_proc) {
                  int global_patch_id =
                     getGlobalPatchNumber(hierarchy, ln, pn);
                  for (std::list<VisItItem>::iterator ipi(d_plot_items.begin());
                       ipi != d_plot_items.end(); ipi++) {
                     for (comp = 0; comp < ipi->d_depth; comp++) {
                        ipi->d_master_min_max[comp][global_patch_id] =
                           buf[item_ctr];
                        item_ctr++;
                     }
                  }  // variables
               } // patch from sending proc?
            } // patches
         } // levels
      } // while msgs_recvd < working procs

      if (d_number_working_slaves > 0) {
         delete[] buf;
      }

   } // proc == SPHERE_MASTER?

}


/*
 *************************************************************************
 *
 * Create a 2D integer array entry in an HDF database with the specified
 * key name.  The array type is based on the hdf type H5T_NATIVE_INT.
 *
 *************************************************************************
 */

void
SphereDataWriter::HDFputIntegerArray2D(
   const std::string& key,
   const int* data,
   const int nelements0,
   const int nelements1,
   const hid_t group_id)
{
   TBOX_ASSERT(!key.empty());
   TBOX_ASSERT(data != 0);
   TBOX_ASSERT((nelements0 > 0) && (nelements1 > 0));

   herr_t errf;
   if ((nelements0 > 0) && (nelements1 > 0)) {
      hsize_t dim[] = { static_cast<hsize_t>(nelements0), static_cast<hsize_t>(nelements1) };
      hid_t space = H5Screate_simple(2, dim, 0);

      TBOX_ASSERT(space >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            H5T_NATIVE_INT,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
#else
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            H5T_NATIVE_INT,
            space,
            H5P_DEFAULT);
#endif

      TBOX_ASSERT(dataset >= 0);

      errf = H5Dwrite(dataset,
            H5T_NATIVE_INT,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data);
      TBOX_ASSERT(errf >= 0);
      NULL_USE(errf);

      errf = H5Sclose(space);

      TBOX_ASSERT(errf >= 0);

      errf = H5Dclose(dataset);
      TBOX_ASSERT(errf >= 0);

   } else {
      TBOX_ERROR("SphereDataWriter::HDFputIntegerArray2D()"
         << "\n    data writer with name " << d_object_name
         << "\n    Attempt to put zero-length array with key = "
         << key << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Create a 2D double array entry in an HDF database with the specified
 * key name.  The array type is based on the hdf type H5T_NATIVE_DOUBLE.
 *
 *************************************************************************
 */

void
SphereDataWriter::HDFputDoubleArray2D(
   const std::string& key,
   const double* data,
   const int nelements0,
   const int nelements1,
   const hid_t group_id)
{

   TBOX_ASSERT(!key.empty());
   TBOX_ASSERT(data != 0);
   TBOX_ASSERT((nelements0 > 0) && (nelements1 > 0));

   herr_t errf;
   if ((nelements0 > 0) && (nelements1 > 0)) {
      hsize_t dim[] = { static_cast<hsize_t>(nelements0), static_cast<hsize_t>(nelements1) };
      hid_t space = H5Screate_simple(2, dim, 0);

      TBOX_ASSERT(space >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            H5T_NATIVE_DOUBLE,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
#else
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            H5T_NATIVE_DOUBLE,
            space,
            H5P_DEFAULT);
#endif

      TBOX_ASSERT(dataset >= 0);

      errf = H5Dwrite(dataset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data);

      TBOX_ASSERT(errf >= 0);
      NULL_USE(errf);

      errf = H5Sclose(space);
      TBOX_ASSERT(errf >= 0);

      errf = H5Dclose(dataset);
      TBOX_ASSERT(errf >= 0);

   } else {
      TBOX_ERROR("SphereDataWriter::HDFputDoubleArray2D()"
         << "\n    data writer with name " << d_object_name
         << "\n    Attempt to put zero-length array with key = "
         << key << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Create an array of patch extent (pe) structs in an HDF database
 * with the specified key name.
 *
 *************************************************************************
 */

void
SphereDataWriter::HDFputPatchExtentsStructArray(
   const std::string& key,
   const patchExtentsStruct* data,
   const int nelements,
   const hid_t group_id)
{
   TBOX_ASSERT(!key.empty());
   TBOX_ASSERT(data != 0);
   TBOX_ASSERT(nelements > 0);

   herr_t errf;
   if (nelements > 0) {
      hid_t space;
      hsize_t dim[1];
      dim[0] = nelements;
      space = H5Screate_simple(1, dim, 0);
      TBOX_ASSERT(space >= 0);

      hid_t pe_id = H5Tcreate(H5T_COMPOUND, sizeof(patchExtentsStruct));
      TBOX_ASSERT(pe_id >= 0);

      hsize_t dim1[1];
      dim1[0] = SPHERE_FIXED_DIM;
#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t intXdType = H5Tarray_create(H5T_NATIVE_INT, 1, dim1);
#else
      hid_t intXdType = H5Tarray_create(H5T_NATIVE_INT, 1, dim1, 0);
#endif
      TBOX_ASSERT(intXdType >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t doubleXdType = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dim1);
#else
      hid_t doubleXdType = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dim1, 0);
#endif
      TBOX_ASSERT(doubleXdType >= 0);

      errf = H5Tinsert(pe_id,
            "lower",
            HOFFSET(patchExtentsStruct, lower),
            intXdType);
      TBOX_ASSERT(errf >= 0);
      NULL_USE(errf);

      errf = H5Tinsert(pe_id,
            "upper",
            HOFFSET(patchExtentsStruct, upper),
            intXdType);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(pe_id,
            "xlo",
            HOFFSET(patchExtentsStruct, xlo),
            doubleXdType);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(pe_id,
            "xup",
            HOFFSET(patchExtentsStruct, xhi),
            doubleXdType);
      TBOX_ASSERT(errf >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            pe_id,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
#else
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            pe_id,
            space,
            H5P_DEFAULT);
#endif
      TBOX_ASSERT(dataset >= 0);

      errf = H5Dwrite(dataset,
            pe_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data);
      TBOX_ASSERT(errf >= 0);

      errf = H5Sclose(space);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tclose(pe_id);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tclose(intXdType);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tclose(doubleXdType);
      TBOX_ASSERT(errf >= 0);

      errf = H5Dclose(dataset);
      TBOX_ASSERT(errf >= 0);

   } else {
      TBOX_ERROR("SphereDataWriter::HDFputPatchExtentsStructArray()"
         << "\n    Attempt to put zero-length array with key = "
         << key << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Create patch map for each of the patch extents HDF entries
 * with the specified key name.
 *
 *************************************************************************
 */

void
SphereDataWriter::HDFputPatchMapStructArray(
   const std::string& key,
   const patchMapStruct* data,
   const int nelements,
   const hid_t group_id)
{
   TBOX_ASSERT(!key.empty());
   TBOX_ASSERT(data != 0);
   TBOX_ASSERT(nelements > 0);

   herr_t errf;
   if (nelements > 0) {
      hid_t space;
      hsize_t dim[1];
      dim[0] = nelements;
      space = H5Screate_simple(1, dim, 0);
      TBOX_ASSERT(space >= 0);

      hid_t pm_id = H5Tcreate(H5T_COMPOUND, sizeof(patchMapStruct));
      TBOX_ASSERT(pm_id >= 0);

      errf = H5Tinsert(pm_id,
            "processor_number",
            HOFFSET(patchMapStruct, processor_number),
            H5T_NATIVE_INT);
      TBOX_ASSERT(errf >= 0);
      NULL_USE(errf);

      errf = H5Tinsert(pm_id,
            "file_cluster_number",
            HOFFSET(patchMapStruct, file_cluster_number),
            H5T_NATIVE_INT);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(pm_id,
            "level_number",
            HOFFSET(patchMapStruct, level_number),
            H5T_NATIVE_INT);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(pm_id,
            "patch_number",
            HOFFSET(patchMapStruct, patch_number),
            H5T_NATIVE_INT);
      TBOX_ASSERT(errf >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            pm_id,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
#else
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            pm_id,
            space,
            H5P_DEFAULT);
#endif
      TBOX_ASSERT(dataset >= 0);
      errf = H5Dwrite(dataset,
            pm_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data);

      TBOX_ASSERT(errf >= 0);

      errf = H5Sclose(space);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tclose(pm_id);
      TBOX_ASSERT(errf >= 0);

      errf = H5Dclose(dataset);
      TBOX_ASSERT(errf >= 0);

   } else {
      TBOX_ERROR("SphereDataWriter::HDFputPatchMapStructArray()"
         << "\n    Attempt to put zero-length array with key = "
         << key << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Create an array of max-min double (mm) structs an HDF database with
 * the specified key name.
 *
 *************************************************************************
 */
void
SphereDataWriter::HDFputPatchMinMaxStructArray(
   const std::string& key,
   const patchMinMaxStruct* data,
   const int nelements,
   const hid_t group_id)
{
   TBOX_ASSERT(!key.empty());
   TBOX_ASSERT(data != 0);
   TBOX_ASSERT(nelements > 0);

   herr_t errf;
   if (nelements > 0) {
      hid_t space;
      hsize_t dim[1];
      dim[0] = nelements;
      space = H5Screate_simple(1, dim, 0);
      TBOX_ASSERT(space >= 0);

      hid_t s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(patchMinMaxStruct));
      TBOX_ASSERT(s1_tid >= 0);

      errf = H5Tinsert(s1_tid,
            "data_is_defined",
            HOFFSET(patchMinMaxStruct, patch_data_on_disk),
            H5T_NATIVE_CHAR);
      TBOX_ASSERT(errf >= 0);
      NULL_USE(errf);

      errf = H5Tinsert(s1_tid,
            "material_composition_flag",
            HOFFSET(patchMinMaxStruct, material_composition_code),
            H5T_NATIVE_INT);

      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(s1_tid,
            "species_composition_flag",
            HOFFSET(patchMinMaxStruct, species_composition_code),
            H5T_NATIVE_INT);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(s1_tid,
            "min",
            HOFFSET(patchMinMaxStruct, min),
            H5T_NATIVE_DOUBLE);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tinsert(s1_tid,
            "max",
            HOFFSET(patchMinMaxStruct, max),
            H5T_NATIVE_DOUBLE);
      TBOX_ASSERT(errf >= 0);

#if (H5_VERS_MAJOR > 1) || ((H5_VERS_MAJOR == 1) && (H5_VERS_MINOR > 6))
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            s1_tid,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);
#else
      hid_t dataset = H5Dcreate(group_id,
            key.c_str(),
            s1_tid,
            space,
            H5P_DEFAULT);
#endif
      TBOX_ASSERT(dataset >= 0);

      errf = H5Dwrite(dataset,
            s1_tid,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data);
      TBOX_ASSERT(errf >= 0);

      errf = H5Sclose(space);
      TBOX_ASSERT(errf >= 0);

      errf = H5Tclose(s1_tid);
      TBOX_ASSERT(errf >= 0);

      errf = H5Dclose(dataset);
      TBOX_ASSERT(errf >= 0);

   } else {
      TBOX_ERROR("SphereDataWriter::HDFputPatchMinMaxStructArray()"
         << "\n    data writer with name " << d_object_name
         << "\n    Attempt to put zero-length array with key = "
         << key << std::endl);
   }
}

/*
 *************************************************************************
 *
 * Dump plotitem fields for debugging purposes
 *
 *************************************************************************
 */
void
SphereDataWriter::dumpItem(
   VisItItem& plotitem,
   std::ostream& os) const
{
   os << "d_var_name: " << plotitem.d_var_name << "\n";
   std::string type;
   if (plotitem.d_var_type == SPHERE_SCALAR) {
      type = "SCALAR";
   } else if (plotitem.d_var_type == SPHERE_VECTOR) {
      type = "VECTOR";
   } else if (plotitem.d_var_type == SPHERE_TENSOR) {
      type = "TENSOR";
   }
   os << "d_var_type: " << type << "\n";
   std::string data_type;
   if (plotitem.d_var_data_type == SPHERE_DOUBLE) {
      data_type = "DOUBLE";
   } else if (plotitem.d_var_data_type == SPHERE_FLOAT) {
      data_type = "FLOAT";
   } else if (plotitem.d_var_data_type == SPHERE_INT) {
      data_type = "INT";
   }
   os << "d_var_data_type: " << data_type << "\n";
   std::string cent;
   if (plotitem.d_var_centering == SPHERE_CELL) {
      cent = "CELL";
   } else if (plotitem.d_var_centering == SPHERE_UNKNOWN_CELL) {
      cent = "UNKNOWN_CELL";
   } else if (plotitem.d_var_centering == SPHERE_NODE) {
      cent = "NODE";
   } else if (plotitem.d_var_centering == SPHERE_UNKNOWN_NODE) {
      cent = "SPHERE_UNKNOWN_NODE";
   }
   os << "d_var_centering: " << cent << "\n";
   os << "d_patch_data_index: " << plotitem.d_patch_data_index << "\n";
   os << "d_depth: " << plotitem.d_depth << "\n";
   os << "d_start_depth_index: " << plotitem.d_start_depth_index << "\n";
   os << "d_scale_factor: " << plotitem.d_scale_factor << "\n";
   
   int i;
   for (i = 0; i < plotitem.d_depth; i++) {
      os << "   comp_name[" << i << "]: "
         << plotitem.d_visit_var_name[i] << "\n";
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

#endif
