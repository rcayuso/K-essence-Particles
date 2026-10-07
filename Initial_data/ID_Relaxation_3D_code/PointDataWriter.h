/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2013 Lawrence Livermore National Security, LLC
 * Description:   Writes data files for visualization by VisIt
 *
 ************************************************************************/

#ifndef included_appu_PointDataWriter
#define included_appu_PointDataWriter

#include "SAMRAI/SAMRAI_config.h"

/*
 ************************************************************************
 *  THIS CLASS WILL BE UNDEFINED IF THE LIBRARY IS BUILT WITHOUT HDF5
 ************************************************************************
 */
#include "SAMRAI/hier/PatchData.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/tbox/IOStream.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/Timer.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include <string>
#include <list>
#include <vector>

using namespace SAMRAI;

class PointDataWriter
{
public:
   /*
    * Constructor
    */
   PointDataWriter(
      const std::vector<double> coordinates,
      const std::string& object_name,
      const std::string& dump_directory_name,
      int number_procs_per_file = 1,
      bool is_multiblock = false);

   /*
    * The destructor
    */
   ~PointDataWriter();


   /*
    * This method registers a variable with the data writer.
    */
   void
   registerPlotQuantity(
      const std::string& variable_name,
      const std::string& variable_type,
      const int patch_data_index,
      const int start_depth_index = 0,
      const double scale_factor = 1.0,
      const std::string& variable_centering = "UNKNOWN");

   /*
    * This method resets the patch_data_index, and/or
    * the depth_index, at a specific level, of a previously registered
    * plot variable.
    */
   void
   resetLevelPlotQuantity(
      const std::string& variable_name,
      const int level_number,
      const int patch_data_index,
      const int start_depth_index = 0);

   /*
    * This method causes the data writer to dump all
    * registered data.
    */
   void
   writePlotData(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      int time_step,
      double simulation_time = 0.0);

   /*
    * Returns the object name.
    */
   const std::string&
   getObjectName() const
   {
      return d_object_name;
   }
   
    /*
    * Sets the mesh variable to use as a mask in multi-level 
    */
   static void setMaskVariable(int mask_var) {
        d_mask_id = mask_var;
   }

private:

    static double d_mask_simulation_time;
    static int d_mask_id;

   /*
    * Static integer constant describing version of VisIt Data Writer.
    */
   static const float POINT_DATAWRITER_VERSION_NUMBER;

   /*
    * Static integer constant describing the maximum size of a C char string.
    */
   static const int POINT_NAME_BUFSIZE;

   /*
    * Static integer constant describing undefined index.
    */
   static const int POINT_UNDEFINED_INDEX;

   /*
    * Static integer constant describing process which writes single summary
    * file with information from all processors for parallel runs
    */
   static const int POINT_MASTER;


   /*
    * hier::Variable type:
    *   SCALAR - scalar plot variable (depth = 1)
    *   VECTOR - vector plot variable (depth = dim)
    *   TENSOR - tensor plot variable (depth = dim*dim)
    */
   enum variable_type { POINT_SCALAR = 0,
                        POINT_VECTOR = 1,
                        POINT_TENSOR = 2 };

   /*
    * hier::Variable data type  - float, double, integer
    */
   enum variable_data_type { POINT_INT = 3,
                             POINT_FLOAT = 4,
                             POINT_DOUBLE = 5,
                             POINT_DATA_TYPE_BAD = 990 };

   /*
    * hier::Variable centering:
    *   CELL         - standard cell centered
    *   NODE         - standard node centered
    *   UNKNOWN_CELL - unknown type, cast to cell centered
    *   UNKNOWN_NODE - unknown type, cast to node centered
    */
   enum variable_centering { POINT_CELL = 6,
                             POINT_NODE = 7,
                             POINT_UNKNOWN_CELL = 8,
                             POINT_UNKNOWN_NODE = 9,
                             POINT_CENTERING_BAD = 991 };

   /*
    * Grid type:
    *   CARTESIAN - standard cartesian grid
    *   DEFORMED  - node centered grid where nodes may be deformed
    *               (e.g. sometimes called curvilinear)
    */
   enum grid_type { POINT_CARTESIAN = 10,
                    POINT_DEFORMED = 11 };

   /*
    * The following structure is used to store data about each item
    * to be written to a plot file.
    */
   struct VisItItem {

      /*
       * Standard information (user supplied)
       */
      std::string d_var_name;
      variable_type d_var_type;
      variable_centering d_var_centering;
      int d_patch_data_index;

      /*
       * Standard information (writer generated)
       */
      int d_depth;
      variable_data_type d_var_data_type;
      std::vector<std::string> d_visit_var_name;
      std::vector<double> d_value;

   };

   /*
    * Utility routine to initialize a standard variable for
    * plotting based on user input.  Derived, coordinate, material,
    * and species data all use this method to initialize the variable
    * and then set specific characteristics in their appropriate
    * register methods.
    */
   void
   initializePlotItem(
      VisItItem& plotitem,
      const std::string& variable_name,
      const std::string& variable_type,
      const int patch_data_index,
      const int start_depth_index,
      const double scale_factor,
      const std::string& variable_centering);

   /*
    * Utility routine to reset a variable by level for plotting
    * (either vector or scalar variable).
    */
   void
   resetLevelPlotItem(
      const std::string& variable_name,
      int level_number,
      int patch_data_array_index,
      int start_depth_index,
      std::string method_name);

   /*
    * Utility routine to reset values from previous calculations
    */
    void clearValues();

   /*
    * Coordinate writing HDF plot files, both data and summary.
    */
   void
   writeFiles(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      double simulation_time);

   /*
    * Performs the calculations
    */
   void
   calculatePoint(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy);

   /*
    * Name of this VisIt data writer object
    */
   std::string d_object_name;

   /*
    * Directory into which VisIt files will be written.
    */
   std::string d_top_level_directory_name;

   /*
    * Time step number (passed in by user).
    */
   int d_time_step_number;

   int d_at_level;
   std::vector<double> d_coordinates;

   /*
    *  tbox::List of scalar and vector variables registered with
    *  VisItDataWriter.
    */
   std::list<VisItItem> d_plot_items;


   //! @brief Timer for writePlotData().
   static std::shared_ptr<tbox::Timer> t_write_plot_data;

   /*!
    * @brief Initialize static objects and register shutdown routine.
    *
    * Only called by StartupShutdownManager.
    */
   static void
   initializeCallback()
   {
      t_write_plot_data = tbox::TimerManager::getManager()->getTimer(
         "appu:PointDataWriter::writePlotData()");
   }

   /*!
    * @brief Method registered with ShutdownRegister to cleanup statics.
    *
    * Only called by StartupShutdownManager.
    */
   static void
   finalizeCallback()
   {
      t_write_plot_data.reset();
   }

   /*
    * Static initialization and cleanup handler.
    */
   static tbox::StartupShutdownManager::Handler
      s_initialize_handler;

};

#endif
