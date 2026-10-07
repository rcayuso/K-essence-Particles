#ifndef included_MainRestartData
#define included_MainRestartData

#include "SAMRAI/SAMRAI_config.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/Serializable.h"
#ifndef included_String
#include <string>
#define included_String
#endif

using namespace std;
using namespace SAMRAI;

/**
 * Class MainRestartData is a concrete subclass of tbox::Serializable that is 
 * used for storing and accessing the data in main that is necessary for 
 * restart.
 */

class MainRestartData: public tbox::Serializable
{
public:
   /**
    * The constructor for the serializable base class does nothing interesting.
    */
   MainRestartData(const string& object_name,
                   std::shared_ptr<tbox::Database>& input_db);

   /**
    * The virtual destructor for the serializable base class does nothing
    * interesting.
    */
   ~MainRestartData();

   /**
    * Returns d_start_time.
    */
   double getStartTime();

   /**
    * Returns d_loop_time.
    */
   double getLoopTime();

   /**
    * Sets d_loop_time.
    */
   void setLoopTime(const double loop_time);

   /**
    * Returns d_iteration_number.
    */
   int getIterationNumber();

   /**
    * Sets d_iteration_number.
    */
   void setIterationNumber(const int iter_num);

	/**
	 * Returns d_next_mesh_dump_iteration.
	 */
	int getNextMeshDumpIteration();

	/**
	 * Sets d_next_mesh_dump_iteration.
	 */
	void setNextMeshDumpIteration(int next_mesh_dump_iteration);


	/**
	 * Returns d_next_analysis_dump_iteration.
	 */
	int getNextDumpAnalysisIteration();

	/**
	 * Sets d_next_analysis_dump_iteration
	 */
	void setNextDumpAnalysisIteration(int next_analysis_dump_iteration);


	/**
	 * Returns d_next_slice_dump_iteration.
	 */
	vector<int> getNextSliceDumpIteration();

	/**
	 * Sets d_next_slice_dump_iteration.
	 */
	void setNextSliceDumpIteration(const vector<int> next_slice_dump_iteration);

	/**
	 * Returns d_next_sphere_dump_iteration.
	 */
	vector<int> getNextSphereDumpIteration();

	/**
	 * Sets d_next_sphere_dump_iteration.
	 */
	void setNextSphereDumpIteration(const vector<int> next_sphere_dump_iteration);


	/**
	 * Returns d_next_integration_dump_iteration.
	 */
	vector<int> getNextIntegrationDumpIteration();

	/**
	 * Sets d_next_integration_dump_iteration.
	 */
	void setNextIntegrationDumpIteration(const vector<int> next_integration_dump_iteration);

	/**
	 * Returns d_next_point_dump_iteration.
	 */
	vector<int> getNextPointDumpIteration();

	/**
	 * Sets d_next_point_dump_iteration.
	 */
	void setNextPointDumpIteration(const vector<int> next_point_dump_iteration);



  /**
   * Returns d_current_iteration.
   */
  vector<int> getCurrentIteration();

  /**
   * Sets d_current_iteration.
   */
  void setCurrentIteration(const vector<int> current_iteration);

  /**
   * Returns d_next_console_output_iteration.
   */
  int getNextConsoleOutputIteration();

  /**
   * Sets d_next_console_output_iteration.
   */
  void setNextConsoleOutputIteration(const int next_console_output_iteration);

  /**
   * Returns d_next_timer_output_iteration.
   */
  int getNextTimerOutputIteration();

  /**
   * Sets d_next_timer_output_iteration.
   */
  void setNextTimerOutputIteration(const int next_timer_output_iteration);



  /**
   * Writes out d_max_timesteps, d_start_time, d_end_time,
   * d_regrid_step, d_tag_buffer, d_loop_time, d_iteration_number.
   */
  void putToRestart(const std::shared_ptr<tbox::Database>& db) const;




private:
   /**
    * Reads in max_timesteps, start_time, end_time,
    * regrid_step, tag_buffer from the specified input database.
    * Any values from the input database override values found
    * in the restart database.
    */
   void getFromInput( std::shared_ptr<tbox::Database>& input_db,
                              bool is_from_restart);

   /**
    * Reads in d_max_timesteps, d_start_time, d_end_time,
    * d_regrid_step, d_tag_buffer, d_loop_time, d_iteration_number
    * from the specified restart database.
    */
   void getFromRestart(); 

   double d_start_time;
   double d_loop_time;
   int d_iteration_number;

	vector<int> d_next_slice_dump_iteration;
	vector<int> d_next_sphere_dump_iteration;
	int d_next_mesh_dump_iteration;
	vector<int> d_next_integration_dump_iteration;
	vector<int> d_next_point_dump_iteration;
	int d_next_analysis_dump_iteration;

   vector<int> d_current_iteration;
   int d_next_console_output_iteration;
   int d_next_timer_output_iteration;

   string d_object_name;

};

#endif
