#include "MainRestartData.h"
#include "SAMRAI/tbox/PIO.h"
#include "SAMRAI/tbox/RestartManager.h"
#include "SAMRAI/tbox/Utilities.h"

/*
*************************************************************************
*                                                                       *
* Constructor                                                           *
*                                                                       *
*************************************************************************
*/

MainRestartData::MainRestartData(const string& object_name,std::shared_ptr<tbox::Database>& input_db):d_object_name(object_name)
{
	TBOX_ASSERT(input_db);

	tbox::RestartManager::getManager()->registerRestartItem(d_object_name, this);

	/*
	* Initialize object with data read from given input/restart databases.
	*/
	bool is_from_restart = tbox::RestartManager::getManager()->isFromRestart();
	if (is_from_restart){
		getFromRestart();
	}
	getFromInput(input_db,is_from_restart);

	/* if not starting from restart file, set loop_time and iteration_number */
	if (!is_from_restart) {
		d_loop_time = d_start_time;
		d_iteration_number = 0;
	}
}

/*
*************************************************************************
*                                                                       *
* Destructor                                                            *
*                                                                       *
*************************************************************************
*/

MainRestartData::~MainRestartData()
{
}

/*
*************************************************************************
*                                                                       *
* Accessor methods for data                                             *
*                                                                       *
*************************************************************************
*/

double MainRestartData::getStartTime()
{
	return d_start_time;
}

double MainRestartData::getLoopTime()
{
	return d_loop_time;
}

void MainRestartData::setLoopTime(const double loop_time)
{
	d_loop_time = loop_time;
}

int MainRestartData::getIterationNumber()
{
	return d_iteration_number;
}

void MainRestartData::setIterationNumber(const int iter_num)
{
	d_iteration_number = iter_num;
}

/**
 * Returns d_next_mesh_dump_iteration.
 */
int MainRestartData::getNextMeshDumpIteration()
{
	return d_next_mesh_dump_iteration;
}

/**
 * Sets d_next_mesh_dump_iteration.
 */
void MainRestartData::setNextMeshDumpIteration(const int next_mesh_dump_iteration)
{
	d_next_mesh_dump_iteration = next_mesh_dump_iteration;
}


/**
 * Returns d_next_analysis_dump_iteration.
 */
int MainRestartData::getNextDumpAnalysisIteration()
{
	return d_next_analysis_dump_iteration;
}

/**
 * Sets d_next_analysis_dump_iteration
 */
void MainRestartData::setNextDumpAnalysisIteration(int next_analysis_dump_iteration)
{
	d_next_analysis_dump_iteration = next_analysis_dump_iteration;
}


/**
 * Returns d_next_slice_dump_iteration.
 */
vector<int> MainRestartData::getNextSliceDumpIteration()
{
	return d_next_slice_dump_iteration;
}

/**
 * Sets d_next_slice_dump_iteration.
 */
void MainRestartData::setNextSliceDumpIteration(const vector<int> next_slice_dump_iteration)
{
	d_next_slice_dump_iteration = next_slice_dump_iteration;
}

/**
 * Returns d_next_sphere_dump_iteration.
 */
vector<int> MainRestartData::getNextSphereDumpIteration()
{
	return d_next_sphere_dump_iteration;
}

/**
 * Sets d_next_sphere_dump_iteration.
 */
void MainRestartData::setNextSphereDumpIteration(const vector<int> next_sphere_dump_iteration)
{
	d_next_sphere_dump_iteration = next_sphere_dump_iteration;
}


	/**
 * Returns d_next_integration_dump_iteration.
 */
vector<int> MainRestartData::getNextIntegrationDumpIteration()
{
	return d_next_integration_dump_iteration;
}

/**
 * Sets d_next_integration_dump_iteration.
 */
void MainRestartData::setNextIntegrationDumpIteration(const vector<int> next_integration_dump_iteration)
{
	d_next_integration_dump_iteration = next_integration_dump_iteration;
}

	/**
 * Returns d_next_point_dump_iteration.
 */
vector<int> MainRestartData::getNextPointDumpIteration()
{
	return d_next_point_dump_iteration;
}

/**
 * Sets d_next_point_dump_iteration.
 */
void MainRestartData::setNextPointDumpIteration(const vector<int> next_point_dump_iteration)
{
	d_next_point_dump_iteration = next_point_dump_iteration;
}


/**
* Returns d_current_iteration.
*/
vector<int> MainRestartData::getCurrentIteration()
{
	return d_current_iteration;
}

/**
* Sets d_current_iteration.
*/
void MainRestartData::setCurrentIteration(const vector<int> current_iteration)
{
	d_current_iteration = current_iteration;
}

/**
* Returns d_next_console_output_iteration.
*/
int MainRestartData::getNextConsoleOutputIteration()
{
	return d_next_console_output_iteration;
}

/**
* Sets d_next_console_output_iteration.
*/
void MainRestartData::setNextConsoleOutputIteration(const int next_console_output_iteration)
{
	d_next_console_output_iteration = next_console_output_iteration;
}

/**
* Returns d_next_timer_output_iteration.
*/
int MainRestartData::getNextTimerOutputIteration()
{
	return d_next_timer_output_iteration;
}

/**
* Sets d_next_timer_output_iteration.
*/
void MainRestartData::setNextTimerOutputIteration(const int next_timer_output_iteration)
{
	d_next_timer_output_iteration = next_timer_output_iteration;
}



/*
*************************************************************************
*                                                                       *
* tbox::Database input/output methods                                         *
*                                                                       *
*************************************************************************
*/
void MainRestartData::putToRestart(const std::shared_ptr<tbox::Database>& db) const
{
	TBOX_ASSERT(db);

	db->putDouble("d_start_time",d_start_time);
	db->putDouble("d_loop_time",d_loop_time);
	db->putInteger("d_iteration_number",d_iteration_number);
	db->putInteger("d_next_console_output_iteration", d_next_console_output_iteration);
	db->putInteger("d_next_timer_output_iteration", d_next_timer_output_iteration);
	if (d_next_slice_dump_iteration.size() > 0) {
		db->putIntegerVector("d_next_slice_dump_iteration",d_next_slice_dump_iteration);
	}
	if (d_next_sphere_dump_iteration.size() > 0) {
		db->putIntegerVector("d_next_sphere_dump_iteration",d_next_sphere_dump_iteration);
	}
	db->putInteger("d_next_mesh_dump_iteration",d_next_mesh_dump_iteration);
	if (d_next_integration_dump_iteration.size() > 0) {
		db->putIntegerVector("d_next_integration_dump_iteration",d_next_integration_dump_iteration);
	}
	if (d_next_point_dump_iteration.size() > 0) {
		db->putIntegerVector("d_next_point_dump_iteration",d_next_point_dump_iteration);
	}
	db->putIntegerVector("d_current_iteration",d_current_iteration);
	db->putInteger("d_next_analysis_dump_iteration",d_next_analysis_dump_iteration);

}

void MainRestartData::getFromInput( std::shared_ptr<tbox::Database>& input_db, bool is_from_restart)
{
   TBOX_ASSERT(input_db);

	d_start_time = 0.0;
}

void MainRestartData::getFromRestart()
{
	std::shared_ptr<tbox::Database> root_db(tbox::RestartManager::getManager()->getRootDatabase());

   	if (!root_db->isDatabase(d_object_name)) {
      		TBOX_ERROR("Restart database corresponding to " << d_object_name << " not found in the restart file.");
   	}
   	std::shared_ptr<tbox::Database> restart_db(root_db->getDatabase(d_object_name));

	d_start_time = restart_db->getDouble("d_start_time");
	d_loop_time = restart_db->getDouble("d_loop_time");
	d_iteration_number = restart_db->getInteger("d_iteration_number");
	d_next_console_output_iteration = restart_db->getInteger("d_next_console_output_iteration");
	d_next_timer_output_iteration = restart_db->getInteger("d_next_timer_output_iteration");
	if (restart_db->keyExists("d_next_slice_dump_iteration")) {
		d_next_slice_dump_iteration = restart_db->getIntegerVector("d_next_slice_dump_iteration");
	}
	if (restart_db->keyExists("d_next_sphere_dump_iteration")) {
		d_next_sphere_dump_iteration = restart_db->getIntegerVector("d_next_sphere_dump_iteration");
	}
	d_next_mesh_dump_iteration = restart_db->getInteger("d_next_mesh_dump_iteration");
	if (restart_db->keyExists("d_next_integration_dump_iteration")) {
		d_next_integration_dump_iteration = restart_db->getIntegerVector("d_next_integration_dump_iteration");
	}
	if (restart_db->keyExists("d_next_point_dump_iteration")) {
		d_next_point_dump_iteration = restart_db->getIntegerVector("d_next_point_dump_iteration");
	}
	d_current_iteration = restart_db->getIntegerVector("d_current_iteration");
	d_next_analysis_dump_iteration = restart_db->getInteger("d_next_analysis_dump_iteration");

}
