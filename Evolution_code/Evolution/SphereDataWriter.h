#ifndef included_appu_SphereDataWriter
#define included_appu_SphereDataWriter

#include "SAMRAI/SAMRAI_config.h"

/*
 ************************************************************************
 *  THIS CLASS WILL BE UNDEFINED IF THE LIBRARY IS BUILT WITHOUT HDF5
 ************************************************************************
 */
#ifdef HAVE_HDF5

#include "SAMRAI/appu/VisDerivedDataStrategy.h"
#include "SAMRAI/appu/VisMaterialsDataStrategy.h"
#include "SAMRAI/hier/PatchData.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/tbox/IOStream.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/HDFDatabase.h"
#include "SAMRAI/tbox/Timer.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include <string>
#include <list>
#include <vector>

using namespace SAMRAI;

/*!
 * @brief Class SphereDataWriter (SDW) is used by SAMRAI-based
 * application codes to generate VisIt data files.  VisIt provides a
 * wide range of visualization and post-processing capabilities.  This
 * class supports both cell-centered and node-centered 3D AMR
 * data where the underlying data type is either double, float, or
 * int. The writer creates a slice (plane) from the 3D data which is dump.
 *
 * After creating a SDW object, the data items to be dumped for
 * plotting must be registered with that object. Thereafter, the
 * object can be used to generate a series of visualization dump files
 * during the execution of an application.  The dumps will include all
 * data items registered with the SDW.
 *
 *
 * The SDW requires the compilation of SAMRAI with the HDF5 library.
 *
 *
 * A brief summary of the basic steps typically involved in using the
 * SDW are:
 *
 *    -  Create a Sphere data writer object, specifying a name for the
 *       object and the name of the directory to contain the visit dump
 *       files.  The axis to which the plane is normal and the distance to axis origin
 *       must be defined. An optional argument number_procs_per_file, applicable
 *       to parallel runs, sets the number of processors that share a
 *       common dump file.  This can reduce parallel I/O contention.
 *       The default value of this arg is 1.  If the value specified
 *       is greater than the number of processors, then all processors
 *       share a single dump file.
 *
 *    - Register hierarchy variable data fields using
 *      registerPlotQuantity().  All variables require a
 *      string identifier and an index into the patch data array on the
 *      AMR hierarchy. 
 *
 *    - The resetLevelPlotQuantity() method is provided for cases when a
 *      variable lives at different patch data indices on different
 *      levels.  Invoking this method redefines the patch data array
 *      index for a given variable on a given level that will be
 *      written to a visit dump file.  Before this method is called,
 *      the variable must be registered using registerPlotQuantity() method.
 *
 *    - The writer will generate VisIt dump files in 2D when the method
 *      writePlotData() is called.  Minimally, only a hierarchy and the
 *      time step number is needed.  A simulation time can also be
 *      specified which will be included as part of the file
 *      information in the dump
 *
 */

class SphereDataWriter
{
public:
   /*!
    * @brief The constructor initializes the Sphere data writer to a
    * default state.
    *
    * The object_name argument is used primarily for error reporting.
    * The dump_directory_name argument is the name of the directory
    * which will contain the visit dump files.  The directory name may
    * include a path.  If the dump directory or any intermediate
    * directories in the path do not exist, they will be created. 
    * The plane_normal_axis sets the axis whose normal is used to create
    * the plane. The distance_to_origin parameter is used to position the plane
    * in the axis in domain units. The
    * optional number_procs_per_file argument is applicable to
    * parallel runs and specifies the number of processors that share
    * a common dump file; the default value is 1. If the specified
    * number_procs_per_file is greater than the number of processors,
    * then all processors share a single vis dump file.  Reducing the
    * number of files written may reduce parallel I/O contention and
    * thus improve I/O efficiency.  The optional argument is_multiblock
    * defaults to false.  It must be set to true for problems on multiblock
    * domains, and left false in all other cases.
    *
    * Before the data writer object can be used for dumping VisIt
    * data, the variables must be registered.
    */
   SphereDataWriter(
      const std::string& object_name,
      const std::string& dump_directory_name,
      const double radius,
      const std::vector<double> center,
      const std::vector<int> resolution);

   /*!
    * @brief The destructor for a SphereDataWriter object.
    *
    * The destructor for the writer does nothing interesting.
    */
   ~SphereDataWriter();


   /*!
    * @brief This method registers a variable with the Sphere data writer.
    *
    * Each plot quantity requires a variable name, which is what VisIt
    * will label the plotted quantity.  The variable type is a string
    * specifying either "SCALAR", "VECTOR", or "TENSOR".  By default, the
    * dimension of a scalar variable is 1, vector is dim, and tensor is
    * dim*dim. The integer patch data array index and optional depth
    * index indicate where the data may be found on patches in the hierarchy.
    *
    * A number of optional parameters may be used to further specify
    * characteristics of the plotted variable. The start depth index
    * allows subsets of variables with depth greater than than the supplied
    * type (scalar, vector, tensor) to be specified. For example, a single
    * depth index of a variable with depth greater than one may be registered
    * as a scalar.  A scale factor may be specified such that each data value
    * is multiplied by this factor before being written to the file. Finally,
    * Finally, the variable centering may be specified for data that is not
    * standard CELL or NODE centered types. By default, the writer will set
    * the centering according to the type of data in the supplied patch data
    * index. It will revert to the supplied type only if it is unable to
    * determine the type from the index.
    *
    * Data does not need to exist on all patches or all levels.
    *
    * An error results and the program will halt if:
    *   - a variable was previously registered with the same name.
    *   - the variable type is not "SCALAR", "VECTOR", or "TENSOR"
    *   - the patch data factory referred to by the patch data array
    *     index is null.
    *   - the start depth index is invalid.
    *   - the supplied variable centering is not "CELL" or "NODE".
    *
    * @param variable_name name of variable.
    * @param variable_type "SCALAR", "VECTOR", "TENSOR"
    * @param patch_data_index patch data descriptor id
    * @param start_depth_index (optional) zero by default; may specify
    *    starting index if patch_data_id has depth greater than
    *    the supplied variable_type
    * @param scale_factor (optional) scale factor for data
    * @param variable_centering (optional) "CELL" or "NODE" - used
    *    only when data being registered is not standard cell or
    *    node type.
    *
    * @pre !variable_name.empty()
    * @pre !variable_type.empty()
    * @pre patch_data_index >= -1
    * @pre start_depth_index >= 0
    */
   void
   registerPlotQuantity(
      const std::string& variable_name,
      const std::string& variable_type,
      const int patch_data_index,
      const int start_depth_index = 0,
      const double scale_factor = 1.0,
      const std::string& variable_centering = "UNKNOWN");

   /*!
    * @brief This method resets the patch_data_index, and/or
    * the depth_index, at a specific level, of a previously registered
    * plot variable.
    *
    * The change redefines the patch data object written to the plot
    * file on the specified level to the data at the new patch data
    * array index / depth index.  This method is used when a
    * particular variable lives at different patch data slots
    * on different hierarchy levels.  For example, suppose a
    * variable lives at a patch data array index on every level except
    * the finest hierarchy level, where it lives at a different index.
    * First, the variable must be registered using
    * registerPlotQuantity().  Second, the patch data index for
    * the finest hierarchy level is reset using this method. When the
    * data is plotted, it will appear on all levels in the
    * hierarchy. The patch data array index must refer to data with
    * the same type (SCALAR, VECTOR, or TENSOR), centering (NODE or CELL),
    * and data type (int, float, double) as the data for which the
    * variable was originally registered.
    *
    * An error results and the program will halt if:
    *   - this variable name was not previously registered.
    *   - the patch data referred to by the patch data array index
    *     is null, or the data type is not the same type as the data
    *     which was originally registered.
    *   - the depth index is invalid.
    *
    * @param variable_name name of variable.
    * @param level_number level number on which data index is being reset.
    * @param patch_data_index new patch data array index.
    * @param start_depth_index (optional) argument indicating the new depth
    *    index.
    *
    * @pre !variable_name.empty()
    * @pre level_number >= 0
    * @pre patch_data_index >= -1
    * @pre start_depth_index >= 0
    */
   void
   resetLevelPlotQuantity(
      const std::string& variable_name,
      const int level_number,
      const int patch_data_index,
      const int start_depth_index = 0);

   /*!
    * @brief This method is used to register node coordinates for
    * deformed structured AMR grids (moving grids).
    *
    * The patch data index must correspond to an dim-dimensional vector
    * that defines the coordinate location ([X,Y] in 2D, [X,Y,Z] in 3D).
    * The data defining the node locations must be node centered.
    *
    * An error results and the program will halt if:
    *   - the patch data array index is invalid.
    *   - the depth of the patch data index is less than dim.
    *
    * If the nodal coordinates are not in a NodeData object on the
    * hierarchy, you can use registerDerivedPlotQuantity() with the
    * variable name of "Coords", the type of "VECTOR" and the
    * variable_centering of "NODE".
    *
    * @param patch_data_index patch data index of the coordinate data.
    * @param start_depth_index (optional) start index for case where
    *    coordinate data is a subcomponent of a larger patch data vector
    *
    * @pre patch_data_index >= -1
    * @pre start_depth_index >= 0
    */
   void
   registerNodeCoordinates(
      const int patch_data_index,
      const int start_depth_index = 0);

   /*!
    * @brief Same as above method, but allows registration of single
    * coordinate for deformed structured AMR grids (moving grids).
    *
    * The coordinate number should be 0, 1, or 2 for X, Y, and Z directions,
    * respectively.  The patch data index must either be a scalar,  or a
    * vector with an appropriate depth index.  A scale factor may be used to
    * scale grid data.
    *
    * If the nodal coordinates are not in a NodeData object on the
    * hierarchy, you can use registerDerivedPlotQuantity() with the
    * variable name of "Coords", the type of "VECTOR" and the
    * variable_centering of "NODE".
    *
    * @param coordinate_number must be 0 or 1 for 2D, or 0,1,2 for 3D.
    * @param patch_data_index patch data index of the coordinate data.
    * @param depth_index (optional) index for case where
    *    coordinate data is a subcomponent of a larger patch data vector
    * @param scale_factor scale factor with which to multiply
    *    coordinate data values
    *
    * @pre (coordinate_number >= 0) && (coordinate_number < d_dim.getValue())
    * @pre patch_data_index >= -1
    * @pre depth_index >= 0
    */
   void
   registerSingleNodeCoordinate(
      const int coordinate_number,
      const int patch_data_index,
      const int depth_index = 0,
      const double scale_factor = 1.0);

   /*!
    * @brief This method causes the Sphere data writer to dump all
    * registered data. The appropriate packing methods will be invoked
    * for material-related data and derived data.
    *
    * The time step number is used as a file name extension for the
    * dump files. It must be non-negative and greater than the
    * previous time step, if any. A simulation time may be provided as
    * an optional argument.  If this time is not specified, a default
    * value of zero is used.  The simulation time can be accessed in
    * VisIt's "File Information" dialog box.
    *
    * An error results and the program will halt if:
    *   - the time step number is <= the previous time step number.
    *   - when assertion checking is active, the hierarchy pointer is null,
    *     the time step is < 0, or the dump directory name string is empty,
    *
    * @param hierarchy A pointer to the patch hierarchy on which the data
    *    to be plotted is defined.
    * @param time_step Non-negative integer value specifying the current
    *    time step number.
    * @param simulation_time Optional argument specifying the double
    *    precision simulation time. Default is 0.0.
    *
    * @pre hierarchy
    * @pre time_step_number >= 0
    * @pre !d_top_level_directory_name.empty()
    */
   void
   writePlotData(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      int time_step,
      double simulation_time = 0.0);

   /*!
    * @brief Set the name of the summary file.
    *
    * This sets the summary file written at each step of the simulation
    * which describes the data contained in the visit files written by
    * each MPI process.  The supplied string is appended with ".samrai"
    * so the actual name of the file will be "<filename>.samrai".  If no
    * alternative name is supplied, by default the summary file used is
    * "summary.samrai".
    *
    * @pre !filename.empty()
    */
   void
   setSummaryFilename(
      std::string& filename)
   {
      TBOX_ASSERT(!filename.empty());
      d_summary_filename = filename + ".samrai";
   }

   /*!
    * @brief Returns the object name.
    *
    * @return The object name.
    */
   const std::string&
   getObjectName() const
   {
      return d_object_name;
   }

private:
   /*
    * Static integer constant describing version of Sphere Data Writer.
    */
   static const float SPHERE_DATAWRITER_VERSION_NUMBER;

   /*
    * Static integer constant describing the maximum number of components
    * ever written.
    */
   static const int SPHERE_MAX_NUMBER_COMPONENTS = 100;

   /*
    * Static integer constant describing the largest base space dimension
    * ever written.
    */
   static const int SPHERE_FIXED_DIM = 3;

   /*
    * Static integer constant describing the maximum size of a C char string.
    */
   static const int SPHERE_NAME_BUFSIZE;

   /*
    * Static integer constant describing undefined index.
    */
   static const int SPHERE_UNDEFINED_INDEX;

   /*
    * Static integer constant describing process which writes single summary
    * file with information from all processors for parallel runs
    */
   static const int SPHERE_MASTER;

   /*
    * Static integer constant describing MPI message tag.
    */
   static const int SPHERE_FILE_CLUSTER_WRITE_BATON;

   /*
    * Static boolean that specifies if the summary file (d_summary_filename)
    * has been opened.
    */
   static bool s_summary_file_opened;

   /*
    * Struct used to gather min/max information, and
    * to track floating point overflows of data.
    */
   struct patchMinMaxStruct {
      double min;
      double max;
      int material_composition_code;
      int species_composition_code;
      char patch_data_on_disk;
   };

   /*
    * Struct to hold patch extents.
    */
   struct patchExtentsStruct {
      int lower[SPHERE_FIXED_DIM];
      int upper[SPHERE_FIXED_DIM];
      double xlo[SPHERE_FIXED_DIM];
      double xhi[SPHERE_FIXED_DIM];
   };

   /*
    * Struct to hold patch processor mapping info.
    */
   struct patchMapStruct {
      int file_cluster_number;
      int processor_number;
      int level_number;
      int patch_number;
   };

   /*
    * hier::Variable type:
    *   SCALAR - scalar plot variable (depth = 1)
    *   VECTOR - vector plot variable (depth = dim)
    *   TENSOR - tensor plot variable (depth = dim*dim)
    */
   enum variable_type { SPHERE_SCALAR = 0,
                        SPHERE_VECTOR = 1,
                        SPHERE_TENSOR = 2 };

   /*
    * hier::Variable data type  - float, double, integer
    */
   enum variable_data_type { SPHERE_INT = 3,
                             SPHERE_FLOAT = 4,
                             SPHERE_DOUBLE = 5,
                             SPHERE_DATA_TYPE_BAD = 990 };

   /*
    * hier::Variable centering:
    *   CELL         - standard cell centered
    *   NODE         - standard node centered
    *   UNKNOWN_CELL - unknown type, cast to cell centered
    *   UNKNOWN_NODE - unknown type, cast to node centered
    */
   enum variable_centering { SPHERE_CELL = 6,
                             SPHERE_NODE = 7,
                             SPHERE_UNKNOWN_CELL = 8,
                             SPHERE_UNKNOWN_NODE = 9,
                             SPHERE_CENTERING_BAD = 991 };

   /*
    * Grid type:
    *   CARTESIAN - standard cartesian grid
    *   DEFORMED  - node centered grid where nodes may be deformed
    *               (e.g. sometimes called curvilinear)
    */
   enum grid_type { SPHERE_CARTESIAN = 10,
                    SPHERE_DEFORMED = 11 };

   /*
    * The following structure is used to store data about each item
    * to be written to a plot file.
    *
    * Standard information (user supplied):
    *   d_var_name - string variable name:
    *   d_var_type - SCALAR, VECTOR, TENSOR
    *   d_var_data_type - INT, FLOAT, DOUBLE
    *   d_var_centering - CELL, NODE, UNKNOWN_CELL, UNKNOWN_NODE
    *   d_patch_data_index - int patch data id
    *   d_depth - int depth of patch data
    *   d_start_depth_index - int starting depth for vector data
    *   d_scale_factor - dbl scaling factor
    *   d_derived_writer - ptr to derived data writer (NULL if not DERIVED)
    *   d_is_material_state_variable - bool, true if mixed state var, in which
    *       case d_derived_writer must be provided and provide
    *       packMixedDerivedDataIntoDoubleBuffer
    *   d_is_species_state_variable - bool, true if mixed state var, in which
    *       case d_derived_writer must be provided and provide
    *       packMixedDerivedDataIntoDoubleBuffer
    *
    * Standard information (writer internal):
    *   d_data_type - INT, FLOAT, DOUBLE
    *   d_visit_var_name - internally maintained visit var name:
    *       for scalar: name[0]  = d_variable_name
    *       for vector: name[0]  = d_variable_name.00,
    *                   name[1]  = d_variable_name.01,
    *                   ..
    *                   name[nn] = d_variable_name.nn
    *   d_master_min_max - ptr to min/max struct on master for each
    *      var.  This is used to gather the min max info for all patches.
    *   d_deformed_coord_id - id of vector defining deformed coordinates
    *   d_coord_scale_factor - scale factor of the different deformed coords
    *   d_level_start_depth_index - int array specifying start depth on
    *                               each level
    *
    * Material information
    *   d_isa_material - boolean specifying if variable is a material
    *   d_material_name - string name of material
    *   d_species_names - string array names of the species
    *   d_materials_writer - ptr to user-supplied materials writer
    *   d_material_name_HDFGroup - hdf group for material
    *
    * Species information
    *   d_isa_species - boolean specifying if variable is a species
    *   d_species_name - string name of species
    *   d_parent_material_pointer - VisItItem ptr to parent material
    *   d_species_HDFGroup - hdf group for species
    *   d_extents_species_HDFGroup - hdf group for species extents
    */
   struct VisItItem {

      /*
       * Standard information (user supplied)
       */
      std::string d_var_name;
      variable_type d_var_type;
      variable_centering d_var_centering;
      int d_patch_data_index;
      int d_start_depth_index;
      double d_scale_factor;
      bool d_is_derived;
      appu::VisDerivedDataStrategy* d_derived_writer;
      bool d_is_deformed_coords;
      // new flag for mixed/clean state variables
      bool d_is_material_state_variable;
      // Do we want to extend this for species, or can that fit into the
      //   material state variable treatment?
      //bool d_is_species_state_variable;

      /*
       * Standard information (writer generated)
       */
      int d_depth;
      variable_data_type d_var_data_type;
      std::vector<std::string> d_visit_var_name;
      struct patchMinMaxStruct *
      d_master_min_max[SPHERE_MAX_NUMBER_COMPONENTS];
      std::vector<int> d_level_patch_data_index;
      std::vector<double> d_coord_scale_factor;

      /*
       * Material information
       */
      bool d_isa_material;
      std::string d_material_name;
      std::vector<std::string> d_species_names;
      appu::VisMaterialsDataStrategy* d_materials_writer;
      /*
       * Species information
       */
      bool d_isa_species;
      std::string d_species_name;
      VisItItem* d_parent_material_pointer;
      std::shared_ptr<tbox::Database> d_species_HDFGroup;
      std::shared_ptr<tbox::Database> d_extents_species_HDFGroup;
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
    * Coordinate writing HDF plot files, both data and summary.
    */
   void
   writeHDFFiles(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      double simulation_time);

   /*
    * Allocate and initialize the min/max structs that hold
    * summary information about each plotted variable.
    */
   void
   initializePlotVariableMinMaxInfo(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy);

   /*
    * Write variable data to HDF file.
    */
   void
   writeVisItVariablesToHDFFile(
      const std::shared_ptr<tbox::Database>& processor_HDFGroup,
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy);

   /*
    * Calculates the value of the spherical projection
    * from a 3D point.
    */
   double
   calculateSphericalPoint(
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
      int levelSize);

   /*
    * Check min/max to make exit cleanly if users data exceeds float
    * min/max values. Otherwise, the writer will dump core when it
    * tries to convert double data to float for writing the vis file.
    */
   void
   checkFloatMinMax(
      const std::string& var_name,
      const double dmin,
      const double dmax);

   /*
    * Convert level number, patch number, to global patch number.
    */
   int
   getGlobalPatchNumber(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      const int level_number,
      const int patch_number);

   /*
    * Write summary data for VisIt to HDF file.
    */
   void
   writeSummaryToHDFFile(
      std::string dump_dir_name,
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      int coarsest_plot_level,
      int finest_plot_level,
      double simulation_time);

   /*
    * Helper method for writeSummaryToHDFFile() method above.
    * Performs tbox::MPI communications to send min/max information for all
    * variables on all patches to the VISIT_MASTER.
    */
   void
   exchangeMinMaxPatchInformation(
      const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
      const int coarsest_plot_level,
      const int finest_plot_level);

   /*
    * Interpolates the spherical point from the indexed cell.
    */
   double
   interpolate(
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
      const hier:: IntVector ghosts);

   /*
    * Create a 2D integer array entry in the database with the specified
    * key name.
    */
   void
   HDFputIntegerArray2D(
      const std::string& key,
      const int* data,
      const int nelements0,
      const int nelements1,
      const hid_t group_id);

   /*
    * Create a 2D double array entry in the database with the specified
    * key name.
    */
   void
   HDFputDoubleArray2D(
      const std::string& key,
      const double* data,
      const int nelements0,
      const int nelements1,
      const hid_t group_id);

   /*
    * Create an array of patch extent structs in the database
    * with the specified key name.
    */
   void
   HDFputPatchExtentsStructArray(
      const std::string& key,
      const patchExtentsStruct* data,
      const int nelements,
      const hid_t group_id);

   /*
    * Create an array of patch map structs in the database
    * with the specified key name.
    */
   void
   HDFputPatchMapStructArray(
      const std::string& key,
      const patchMapStruct* data,
      const int nelements,
      const hid_t group_id);

   /*
    * Create an array of min/max structs in the database with
    * the specified key name.
    */
   void
   HDFputPatchMinMaxStructArray(
      const std::string& key,
      const patchMinMaxStruct* data,
      const int nelements,
      const hid_t group_id);

   /*
    * Dump item fields for debugging purposes.
    */
   void
   dumpItem(
      VisItItem& plotitem,
      std::ostream& os) const;

   /*
    * Dimension of object
    */
   tbox::Dimension d_dim;

   /*
    * Name of this VisIt data writer object
    */
   std::string d_object_name;

   /*
    * Hierarchy level information to write to a file.
    */
   int d_number_levels;

   /*
    * tbox::Array of mesh-scaling ratios from each level to reference level
    * (i.e., coarsest level).
    */
   std::vector<hier::IntVector> d_scaling_ratios;

   /*
    * Directory into which VisIt files will be written.
    */
   std::string d_top_level_directory_name;
   std::string d_current_dump_directory_name;
   std::string d_summary_filename;

   /*
    * Slice information
    */
    double d_radius;
    std::vector<double> d_center;
    std::vector<int> d_resolution;
    double delta_phi;
    double delta_theta;

   /*
    * Grid type - CARTESIAN or DEFORMED.
    */
   grid_type d_grid_type;

   /*
    * Time step number (passed in by user).
    */
   int d_time_step_number;

   /*
    * Number of worker processors that pass info to VISIT_MASTER
    */
   int d_number_working_slaves;

   /*
    * Number of registered VisIt variables, materials, and species.
    * Each regular and derived and variable (i.e. variables registered with
    * a patch data index) are considered visit variables.  If the grid is
    * deformed, the node coordinates registered are also considered
    * visit variables.  Material and species data are NOT counted in
    * d_number_visit_variables because these are supplied by the user. The
    * d_number_visit_variables does not consider whether the plotted variable
    * is scalar, vector, or tensor.  The d_number_visit_variables_plus_depth
    * does consider this, and is the sume of the depths of all the visit
    * variables, again not counting material and species data.
    *
    * The material names are stored in d_materials_names.  Any or all of the
    * materials may have an associated species, and the TOTAL of these is
    * counted in d_number_species.
    */
   int d_number_visit_variables;
   int d_number_visit_variables_plus_depth;
   int d_number_species;

   /*
    * For parallel runs, this array of min/max structs holds the summary
    * information all local patches on workers prior to being sent to
    * master.  It is filled in this order: by level, then by
    * local_patch_number, then by var_item, then by component_number.
    */
   patchMinMaxStruct* d_worker_min_max;
   int d_var_id_ctr;

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
         "appu:SphereDataWriter::writePlotData()");
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
#endif
