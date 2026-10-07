#include "Problem.h"
#include "Functions.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stack>
#include "hdf5.h"
#include <gsl/gsl_rng.h>
#include <random>
#include "SAMRAI/pdat/CellVariable.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/NodeVariable.h"
#include "LagrangianPolynomicRefine.h"


#include "SAMRAI/tbox/Array.h"
#include "SAMRAI/hier/BoundaryBox.h"
#include "SAMRAI/hier/BoxContainer.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"
#include "SAMRAI/hier/VariableDatabase.h"
#include "SAMRAI/hier/PatchDataRestartManager.h"
#include "SAMRAI/hier/Index.h"
#include "SAMRAI/tbox/PIO.h"
#include "SAMRAI/tbox/Utilities.h"
#include "SAMRAI/tbox/Timer.h"
#include "SAMRAI/tbox/TimerManager.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define SIGN(X) (((X) > 0) - ((X) < 0))
#define isEven(a) ((a) % 2 == 0 ? true : false)
#define greaterEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false : (floor(fabs((a) - (b))/1.0E-15) < 1)) || (b)<(a))
#define lessEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false: (floor(fabs((a) - (b))/1.0E-15) < 1)) || (a)<(b))
#define equalsEq(a,b) ((fabs((a) - (b))/1.0E-15 > 10 ? false: (floor(fabs((a) - (b))/1.0E-15) < 1)))
#define reducePrecision(x, p) (floor(((x) * pow(10, (p)) + 0.5)) / pow(10, (p)))

using namespace external;

const gsl_rng_type * T;
gsl_rng *r_var;

//Timers
std::shared_ptr<tbox::Timer> t_step;
std::shared_ptr<tbox::Timer> t_moveParticles;
std::shared_ptr<tbox::Timer> t_output;

inline int Problem::GetExpoBase2(double d)
{
	int i = 0;
	((short *)(&i))[0] = (((short *)(&d))[3] & (short)32752); // _123456789ab____ & 0111111111110000
	return (i >> 4) - 1023;
}

bool	Problem::Equals(double d1, double d2)
{
	if (d1 == d2)
		return true;
	int e1 = GetExpoBase2(d1);
	int e2 = GetExpoBase2(d2);
	int e3 = GetExpoBase2(d1 - d2);
	if ((e3 - e2 < -48) && (e3 - e1 < -48))
		return true;
	return false;
}

/*
 * Constructor of the problem.
 */
Problem::Problem(const string& object_name, const tbox::Dimension& dim, std::shared_ptr<tbox::Database>& database, std::shared_ptr<geom::CartesianGridGeometry >& grid_geom, std::shared_ptr<hier::PatchHierarchy >& patch_hierarchy, MainRestartData& mrd, const double dt, const bool init_from_restart, const int console_output, const int timer_output, const int mesh_output_period, const vector<string> full_mesh_writer_variables, std::shared_ptr<appu::VisItDataWriter>& mesh_data_writer, const vector<int> slicer_output_period, const vector<set<string> > sliceVariables, vector<std::shared_ptr<SlicerDataWriter > >sliceWriters, const vector<int> sphere_output_period, const vector<set<string> > sphereVariables, vector<std::shared_ptr<SphereDataWriter > > sphereWriters, const vector<bool> slicer_analysis_output, const vector<bool> sphere_analysis_output, const vector<int> integration_output_period, const vector<set<string> > integralVariables, vector<std::shared_ptr<IntegrateDataWriter > > integrateDataWriters, const vector<bool> integration_analysis_output, const vector<int> point_output_period, const vector<set<string> > pointVariables, vector<std::shared_ptr<PointDataWriter > > pointDataWriters, const vector<bool> point_analysis_output): 
d_dim(dim), xfer::RefinePatchStrategy(), xfer::CoarsenPatchStrategy(), viz_mesh_dump_interval(mesh_output_period), d_full_mesh_writer_variables(full_mesh_writer_variables.begin(), full_mesh_writer_variables.end()), d_visit_data_writer(mesh_data_writer), d_output_interval(console_output), d_timer_output_interval(timer_output), d_sliceWriters(sliceWriters.begin(), sliceWriters.end()), d_sphereWriters(sphereWriters.begin(), sphereWriters.end()), d_slicer_output_period(slicer_output_period.begin(), slicer_output_period.end()), d_sphere_output_period(sphere_output_period.begin(), sphere_output_period.end()), analysis_slice_dump(slicer_analysis_output.begin(), slicer_analysis_output.end()), analysis_sphere_dump(sphere_analysis_output.begin(), sphere_analysis_output.end()), d_integrateDataWriters(integrateDataWriters.begin(), integrateDataWriters.end()), d_integration_output_period(integration_output_period.begin(), integration_output_period.end()), analysis_integration_dump(integration_analysis_output.begin(), integration_analysis_output.end()), d_pointDataWriters(pointDataWriters.begin(), pointDataWriters.end()), d_point_output_period(point_output_period.begin(), point_output_period.end()), analysis_point_dump(point_analysis_output.begin(), point_analysis_output.end())
{
	//Setup the timers
	t_step = tbox::TimerManager::getManager()->getTimer("Step");
	t_moveParticles = tbox::TimerManager::getManager()->getTimer("Move particles");
	t_output = tbox::TimerManager::getManager()->getTimer("OutputGeneration");

	//Output configuration
	next_console_output = d_output_interval;
	next_timer_output = d_timer_output_interval;

	//Get the object name, the grid geometry and the patch hierarchy
  	d_grid_geometry = grid_geom;
	d_patch_hierarchy = patch_hierarchy;
	d_object_name = object_name;
	d_init_from_restart = init_from_restart;
	initial_dt = dt;

	for (vector<set<string> >::const_iterator it = sliceVariables.begin(); it != sliceVariables.end(); ++it) {
		set<string> vars = *it;
		for (set<string>::const_iterator it2 = vars.begin() ; it2 != vars.end(); ++it2) {
			d_sliceVariables.push_back(vars);
		}
	}
	for (vector<set<string> >::const_iterator it = sphereVariables.begin(); it != sphereVariables.end(); ++it) {
		set<string> vars = *it;
		for (set<string>::const_iterator it2 = vars.begin() ; it2 != vars.end(); ++it2) {
			d_sphereVariables.push_back(vars);
		}
	}
	for (vector<set<string> >::const_iterator it = integralVariables.begin(); it != integralVariables.end(); ++it) {
		set<string> vars = *it;
		for (set<string>::const_iterator it2 = vars.begin() ; it2 != vars.end(); ++it2) {
			d_integralVariables.push_back(vars);
		}
	}
	for (vector<set<string> >::const_iterator it = pointVariables.begin(); it != pointVariables.end(); ++it) {
		set<string> vars = *it;
		for (set<string>::const_iterator it2 = vars.begin() ; it2 != vars.end(); ++it2) {
			d_pointVariables.push_back(vars);
		}
	}


	//Get parameters
    cout<<"Reading parameters"<<endl;
	Phid_y_falloff = database->getDouble("Phid_y_falloff");
	Psisf_falloff = database->getDouble("Psisf_falloff");
	timehyper = database->getDouble("timehyper");
	torbit = database->getDouble("torbit");
	dissipation_factor_Phid_z = database->getDouble("dissipation_factor_Phid_z");
	dissipation_factor_Phid_x = database->getDouble("dissipation_factor_Phid_x");
	massfactor = database->getDouble("massfactor");
	dissipation_factor_Phid_y = database->getDouble("dissipation_factor_Phid_y");
	nu_width = database->getDouble("nu_width");
	dissipation_factor_Psisf = database->getDouble("dissipation_factor_Psisf");
	betax = database->getDouble("betax");
	Nstar = database->getInteger("Nstar");
	Mpl_cte = database->getDouble("Mpl_cte");
	rstart = database->getDouble("rstart");
	tslow2 = database->getDouble("tslow2");
	Phid_z_asymptotic = database->getDouble("Phid_z_asymptotic");
	mu = database->getDouble("mu");
	phi_asymptotic = database->getDouble("phi_asymptotic");
	phi_falloff = database->getDouble("phi_falloff");
	max_error = database->getDouble("max_error");
	Phid_y_asymptotic = database->getDouble("Phid_y_asymptotic");
	dissipation_factor_Psi = database->getDouble("dissipation_factor_Psi");
	zcenter1 = database->getDouble("zcenter1");
	xcenter1 = database->getDouble("xcenter1");
	phi_init = database->getDouble("phi_init");
	rorbit = database->getDouble("rorbit");
	Phid_x_asymptotic = database->getDouble("Phid_x_asymptotic");
	nu_radius = database->getDouble("nu_radius");
	Phid_x_falloff = database->getDouble("Phid_x_falloff");
	sigmax = database->getDouble("sigmax");
	tini = database->getDouble("tini");
	mass = database->getDouble("mass");
	nu = database->getDouble("nu");
	p_sigma = database->getDouble("p_sigma");
	rdonut = database->getDouble("rdonut");
	wd = database->getDouble("wd");
	p_Omega = database->getDouble("p_Omega");
	tend = database->getDouble("tend");
	nu_asymp = database->getDouble("nu_asymp");
	Psisf_asymptotic = database->getDouble("Psisf_asymptotic");
	Phid_z_falloff = database->getDouble("Phid_z_falloff");
	offw = database->getDouble("offw");
	Aphi0 = database->getDouble("Aphi0");
	dissipation_factor_phi = database->getDouble("dissipation_factor_phi");
	Aphi1 = database->getDouble("Aphi1");
	tslow = database->getDouble("tslow");
	sigmaslow = database->getDouble("sigmaslow");
	xf = database->getDouble("xf");
	ycenter1 = database->getDouble("ycenter1");
	gammax = database->getDouble("gammax");
	//Random initialization
	gsl_rng_env_setup();
	//Random for simulation
	const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
	int random_seed = database->getDouble("random_seed")*(mpi.getRank() + 1);
	r_var = gsl_rng_alloc(gsl_rng_ranlxs0);
	gsl_rng_set(r_var, random_seed);
	for (int il = 0; il < d_patch_hierarchy->getMaxNumberOfLevels(); il++) {
		bo_substep_iteration.push_back(0);
	}
	//Initialization from input file
	if (!d_init_from_restart) {
		next_mesh_dump_iteration = viz_mesh_dump_interval;
		for (std::vector<int>::iterator it = d_slicer_output_period.begin(); it != d_slicer_output_period.end(); ++it) {
			next_slice_dump_iteration.push_back((*it));
		}
		for (std::vector<int>::iterator it = d_sphere_output_period.begin(); it != d_sphere_output_period.end(); ++it) {
			next_sphere_dump_iteration.push_back((*it));
		}
		for (std::vector<int>::iterator it = d_integration_output_period.begin(); it != d_integration_output_period.end(); ++it) {
			next_integration_dump_iteration.push_back((*it));
		}
		for (std::vector<int>::iterator it = d_point_output_period.begin(); it != d_point_output_period.end(); ++it) {
			next_point_dump_iteration.push_back((*it));
		}

		//Iteration counter
		for (int il = 0; il < d_patch_hierarchy->getMaxNumberOfLevels(); il++) {
			current_iteration.push_back(0);
		}
	}
	//Initialization from restart file
	else {
		getFromRestart(mrd);
		if (d_slicer_output_period.size() < next_slice_dump_iteration.size()) {
			TBOX_ERROR("Number of slices cannot be reduced after a checkpoint.");
		}
		for (int il = 0; il < d_slicer_output_period.size(); il++) {
			if (il >= next_slice_dump_iteration.size()) {
				next_slice_dump_iteration.push_back(0);
			}
			if (next_slice_dump_iteration[il] == 0 && d_slicer_output_period[il] > 0) {
				next_slice_dump_iteration[il] = current_iteration[d_patch_hierarchy->getNumberOfLevels() - 1] + d_slicer_output_period[il];
			}
		}
		if (d_sphere_output_period.size() < next_sphere_dump_iteration.size()) {
			TBOX_ERROR("Number of spheres cannot be reduced after a checkpoint.");
		}
		for (int il = 0; il < d_sphere_output_period.size(); il++) {
			if (il >= next_sphere_dump_iteration.size()) {
				next_sphere_dump_iteration.push_back(0);
			}
			if (next_sphere_dump_iteration[il] == 0 && d_sphere_output_period[il] > 0) {
				next_sphere_dump_iteration[il] = current_iteration[d_patch_hierarchy->getNumberOfLevels() - 1] + d_sphere_output_period[il];
			}
		}
		if (d_integration_output_period.size() < next_integration_dump_iteration.size()) {
			TBOX_ERROR("Number of integrations cannot be reduced after a checkpoint.");
		}
		for (int il = 0; il < d_integration_output_period.size(); il++) {
			if (il >= next_integration_dump_iteration.size()) {
				next_integration_dump_iteration.push_back(0);
			}
			if (next_integration_dump_iteration[il] == 0 && d_integration_output_period[il] > 0) {
				next_integration_dump_iteration[il] = current_iteration[d_patch_hierarchy->getNumberOfLevels() - 1] + d_integration_output_period[il];
			}
		}
		if (d_point_output_period.size() < next_point_dump_iteration.size()) {
			TBOX_ERROR("Number of point output cannot be reduced after a checkpoint.");
		}
		for (int il = 0; il < d_point_output_period.size(); il++) {
			if (il >= next_point_dump_iteration.size()) {
				next_point_dump_iteration.push_back(0);
			}
			if (next_point_dump_iteration[il] == 0 && d_point_output_period[il] > 0) {
				next_point_dump_iteration[il] = current_iteration[d_patch_hierarchy->getNumberOfLevels() - 1] + d_point_output_period[il];
			}
		}
	}


	//External eos parameters
#ifdef EXTERNAL_EOS

	std::shared_ptr<tbox::Database> external_eos_db = database->getDatabase("external_EOS");
	Commons::ExternalEos::reprimand_eos_type = external_eos_db->getInteger("eos_type");
	Commons::ExternalEos::reprimand_atmo_Ye = external_eos_db->getDouble("atmo_Ye");
	Commons::ExternalEos::reprimand_max_z = external_eos_db->getDouble("max_z");
	Commons::ExternalEos::reprimand_max_b = external_eos_db->getDouble("max_b");
	Commons::ExternalEos::reprimand_c2p_acc = external_eos_db->getDouble("c2p_acc");
	Commons::ExternalEos::reprimand_atmo_rho = external_eos_db->getDouble("atmo_rho");
	Commons::ExternalEos::reprimand_rho_strict = external_eos_db->getDouble("rho_strict");
	Commons::ExternalEos::reprimand_max_rho = external_eos_db->getDouble("max_rho");
	Commons::ExternalEos::reprimand_max_eps = external_eos_db->getDouble("max_eps");
	Commons::ExternalEos::reprimand_gamma_th = external_eos_db->getDouble("gamma_th");
#endif

    	//Subcycling
	d_refinedTimeStepping = false;
	if (database->isString("subcycling")) {
		if (database->getString("subcycling") == "BERGER-OLIGER") {
			d_refinedTimeStepping = true;
		}
	}


	//Regridding options
	d_regridding = false;
	if (database->isDatabase("regridding")) {
		regridding_db = database->getDatabase("regridding");
		d_regridding_buffer = regridding_db->getDouble("regridding_buffer");
		int smallest_patch_size = d_patch_hierarchy->getSmallestPatchSize(0).min();
		for (int il = 1; il < d_patch_hierarchy->getMaxNumberOfLevels(); il++) {
			smallest_patch_size = MIN(smallest_patch_size, d_patch_hierarchy->getSmallestPatchSize(il).min());
		}
		if (d_regridding_buffer > smallest_patch_size) {
			TBOX_ERROR("Error: Regridding_buffer parameter ("<<d_regridding_buffer<<") cannot be greater than smallest_patch_size minimum value("<<smallest_patch_size<<")");
		}
		if (regridding_db->isString("regridding_type")) {
			d_regridding_type = regridding_db->getString("regridding_type");
			d_regridding_min_level = regridding_db->getInteger("regridding_min_level");
			d_regridding_max_level = regridding_db->getInteger("regridding_max_level");
			if (d_regridding_type == "GRADIENT") {
				d_regridding_field = regridding_db->getString("regridding_field");
				d_regridding_compressionFactor = regridding_db->getDouble("regridding_compressionFactor");
				d_regridding_mOffset = regridding_db->getDouble("regridding_mOffset");
				d_regridding = true;
			} else {
				if (d_regridding_type == "FUNCTION") {
					d_regridding_field = regridding_db->getString("regridding_function_field");
					d_regridding_threshold = regridding_db->getDouble("regridding_threshold");
					d_regridding = true;
				} else {
					if (d_regridding_type == "SHADOW") {
						std::string* fields = new std::string[2];
						regridding_db->getStringArray("regridding_fields", fields, 2);
						d_regridding_field = fields[0];
						d_regridding_field_shadow = fields[1];
						d_regridding_error = regridding_db->getDouble("regridding_error");
						d_regridding = true;
						delete[] fields;
					}
				}
			}
		}
	}

	//Stencil of the discretization method
	int maxratio = 1;
	for (int il = 1; il < d_patch_hierarchy->getMaxNumberOfLevels(); il++) {
		const hier::IntVector ratio = d_patch_hierarchy->getRatioToCoarserLevel(il);
		maxratio = MAX(maxratio, ratio.max());
	}
	//Minimum region thickness
	d_regionMinThickness = 3;
	d_ghost_width = 3;

	
	//Register Fields and temporal fields into the variable database
	hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
  	std::shared_ptr<hier::VariableContext> d_cont_curr(vdb->getContext("Current"));
	std::shared_ptr< pdat::CellVariable<int> > mask(std::shared_ptr< pdat::CellVariable<int> >(new pdat::CellVariable<int>(d_dim, "samrai_mask",1)));
	d_mask_id = vdb->registerVariableAndContext(mask ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	IntegrateDataWriter::setMaskVariable(d_mask_id);
	std::shared_ptr< pdat::NodeVariable<double> > interior(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "regridding_value",1)));
	d_interior_regridding_value_id = vdb->registerVariableAndContext(interior ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<int> > nonSync(std::shared_ptr< pdat::NodeVariable<int> >(new pdat::NodeVariable<int>(d_dim, "regridding_tag",1)));
	d_nonSync_regridding_tag_id = vdb->registerVariableAndContext(nonSync ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > interior_i(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "interior_i",1)));
	d_interior_i_id = vdb->registerVariableAndContext(interior_i ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > interior_j(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "interior_j",1)));
	d_interior_j_id = vdb->registerVariableAndContext(interior_j ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > interior_k(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "interior_k",1)));
	d_interior_k_id = vdb->registerVariableAndContext(interior_k ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > FOV_1(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_1",1)));
	d_FOV_1_id = vdb->registerVariableAndContext(FOV_1 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_1_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_xLower(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_xLower",1)));
	d_FOV_xLower_id = vdb->registerVariableAndContext(FOV_xLower ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_xLower_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_xUpper(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_xUpper",1)));
	d_FOV_xUpper_id = vdb->registerVariableAndContext(FOV_xUpper ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_xUpper_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_yLower(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_yLower",1)));
	d_FOV_yLower_id = vdb->registerVariableAndContext(FOV_yLower ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_yLower_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_yUpper(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_yUpper",1)));
	d_FOV_yUpper_id = vdb->registerVariableAndContext(FOV_yUpper ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_yUpper_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_zLower(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_zLower",1)));
	d_FOV_zLower_id = vdb->registerVariableAndContext(FOV_zLower ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_zLower_id);
	std::shared_ptr< pdat::NodeVariable<double> > FOV_zUpper(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "FOV_zUpper",1)));
	d_FOV_zUpper_id = vdb->registerVariableAndContext(FOV_zUpper ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_FOV_zUpper_id);
	std::shared_ptr< pdat::NodeVariable<double> > Psisf(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Psisf",1)));
	d_Psisf_id = vdb->registerVariableAndContext(Psisf ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_Psisf_id);
	std::shared_ptr< pdat::NodeVariable<double> > phi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "phi",1)));
	d_phi_id = vdb->registerVariableAndContext(phi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_phi_id);
	std::shared_ptr< pdat::NodeVariable<double> > Phid_x(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_x",1)));
	d_Phid_x_id = vdb->registerVariableAndContext(Phid_x ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_Phid_x_id);
	std::shared_ptr< pdat::NodeVariable<double> > Phid_y(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_y",1)));
	d_Phid_y_id = vdb->registerVariableAndContext(Phid_y ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_Phid_y_id);
	std::shared_ptr< pdat::NodeVariable<double> > Phid_z(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_z",1)));
	d_Phid_z_id = vdb->registerVariableAndContext(Phid_z ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_Phid_z_id);
	std::shared_ptr< pdat::NodeVariable<double> > Psi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Psi",1)));
	d_Psi_id = vdb->registerVariableAndContext(Psi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_Psi_id);
	std::shared_ptr< pdat::NodeVariable<double> > phi2(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "phi2",1)));
	d_phi2_id = vdb->registerVariableAndContext(phi2 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_phi2_id);
	std::shared_ptr< pdat::NodeVariable<double> > dphi2(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "dphi2",1)));
	d_dphi2_id = vdb->registerVariableAndContext(dphi2 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_dphi2_id);
	std::shared_ptr< pdat::NodeVariable<double> > phi22(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "phi22",1)));
	d_phi22_id = vdb->registerVariableAndContext(phi22 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vxplus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vxplus",1)));
	d_vxplus_id = vdb->registerVariableAndContext(vxplus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vxminus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vxminus",1)));
	d_vxminus_id = vdb->registerVariableAndContext(vxminus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vyplus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vyplus",1)));
	d_vyplus_id = vdb->registerVariableAndContext(vyplus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vyminus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vyminus",1)));
	d_vyminus_id = vdb->registerVariableAndContext(vyminus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vzplus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vzplus",1)));
	d_vzplus_id = vdb->registerVariableAndContext(vzplus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vzminus(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vzminus",1)));
	d_vzminus_id = vdb->registerVariableAndContext(vzminus ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > vorticity_norm(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "vorticity_norm",1)));
	d_vorticity_norm_id = vdb->registerVariableAndContext(vorticity_norm ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > d2tphi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "d2tphi",1)));
	d_d2tphi_id = vdb->registerVariableAndContext(d2tphi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Chi_t(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Chi_t",1)));
	d_Chi_t_id = vdb->registerVariableAndContext(Chi_t ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Chi_x(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Chi_x",1)));
	d_Chi_x_id = vdb->registerVariableAndContext(Chi_x ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Chi_y(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Chi_y",1)));
	d_Chi_y_id = vdb->registerVariableAndContext(Chi_y ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Chi_z(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Chi_z",1)));
	d_Chi_z_id = vdb->registerVariableAndContext(Chi_z ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Chi_norm(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Chi_norm",1)));
	d_Chi_norm_id = vdb->registerVariableAndContext(Chi_norm ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1Psisf(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1Psisf",1)));
	d_rk1Psisf_id = vdb->registerVariableAndContext(rk1Psisf ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1phi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1phi",1)));
	d_rk1phi_id = vdb->registerVariableAndContext(rk1phi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1Phid_x(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1Phid_x",1)));
	d_rk1Phid_x_id = vdb->registerVariableAndContext(rk1Phid_x ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1Phid_y(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1Phid_y",1)));
	d_rk1Phid_y_id = vdb->registerVariableAndContext(rk1Phid_y ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1Phid_z(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1Phid_z",1)));
	d_rk1Phid_z_id = vdb->registerVariableAndContext(rk1Phid_z ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk1Psi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk1Psi",1)));
	d_rk1Psi_id = vdb->registerVariableAndContext(rk1Psi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2Psisf(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2Psisf",1)));
	d_rk2Psisf_id = vdb->registerVariableAndContext(rk2Psisf ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2phi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2phi",1)));
	d_rk2phi_id = vdb->registerVariableAndContext(rk2phi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2Phid_x(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2Phid_x",1)));
	d_rk2Phid_x_id = vdb->registerVariableAndContext(rk2Phid_x ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2Phid_y(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2Phid_y",1)));
	d_rk2Phid_y_id = vdb->registerVariableAndContext(rk2Phid_y ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2Phid_z(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2Phid_z",1)));
	d_rk2Phid_z_id = vdb->registerVariableAndContext(rk2Phid_z ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk2Psi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk2Psi",1)));
	d_rk2Psi_id = vdb->registerVariableAndContext(rk2Psi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3Psisf(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3Psisf",1)));
	d_rk3Psisf_id = vdb->registerVariableAndContext(rk3Psisf ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3phi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3phi",1)));
	d_rk3phi_id = vdb->registerVariableAndContext(rk3phi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3Phid_x(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3Phid_x",1)));
	d_rk3Phid_x_id = vdb->registerVariableAndContext(rk3Phid_x ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3Phid_y(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3Phid_y",1)));
	d_rk3Phid_y_id = vdb->registerVariableAndContext(rk3Phid_y ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3Phid_z(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3Phid_z",1)));
	d_rk3Phid_z_id = vdb->registerVariableAndContext(rk3Phid_z ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > rk3Psi(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "rk3Psi",1)));
	d_rk3Psi_id = vdb->registerVariableAndContext(rk3Psi ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2",1)));
	d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id = vdb->registerVariableAndContext(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id);
	std::shared_ptr< pdat::NodeVariable<double> > d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2",1)));
	d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id = vdb->registerVariableAndContext(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id);
	std::shared_ptr< pdat::NodeVariable<double> > d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2",1)));
	d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id = vdb->registerVariableAndContext(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id);
	std::shared_ptr< pdat::NodeVariable<double> > stalled_1(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "stalled_1",1)));
	d_stalled_1_id = vdb->registerVariableAndContext(stalled_1 ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	hier::PatchDataRestartManager::getManager()->registerPatchDataForRestart(d_stalled_1_id);
	std::shared_ptr< pdat::NodeVariable<double> > Psisf_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Psisf_p",1)));
	d_Psisf_p_id = vdb->registerVariableAndContext(Psisf_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > phi_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "phi_p",1)));
	d_phi_p_id = vdb->registerVariableAndContext(phi_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Phid_x_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_x_p",1)));
	d_Phid_x_p_id = vdb->registerVariableAndContext(Phid_x_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Phid_y_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_y_p",1)));
	d_Phid_y_p_id = vdb->registerVariableAndContext(Phid_y_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Phid_z_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Phid_z_p",1)));
	d_Phid_z_p_id = vdb->registerVariableAndContext(Phid_z_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));
	std::shared_ptr< pdat::NodeVariable<double> > Psi_p(std::shared_ptr< pdat::NodeVariable<double> >(new pdat::NodeVariable<double>(d_dim, "Psi_p",1)));
	d_Psi_p_id = vdb->registerVariableAndContext(Psi_p ,d_cont_curr ,hier::IntVector(d_dim, d_ghost_width));


	//Refine and coarse algorithms

	d_bdry_fill_init = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_post_coarsen = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_fill_advance1 = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_fill_advance7 = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_fill_advance13 = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_fill_advance19 = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_bdry_fill_analysis1 = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_coarsen_algorithm = std::shared_ptr< xfer::CoarsenAlgorithm >(new xfer::CoarsenAlgorithm(d_dim));

	d_mapping_fill = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
    d_tagging_fill = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_fill_new_level    = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());
	d_fill_new_level_aux    = std::shared_ptr< xfer::RefineAlgorithm >(new xfer::RefineAlgorithm());


	//mapping communication

	std::shared_ptr< hier::RefineOperator > refine_operator_map = d_grid_geometry->lookupRefineOperator(interior, "LINEAR_REFINE");
	d_mapping_fill->registerRefine(d_interior_regridding_value_id,d_interior_regridding_value_id,d_interior_regridding_value_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_interior_i_id,d_interior_i_id,d_interior_i_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_interior_j_id,d_interior_j_id,d_interior_j_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_interior_k_id,d_interior_k_id,d_interior_k_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_1_id,d_FOV_1_id,d_FOV_1_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_xLower_id,d_FOV_xLower_id,d_FOV_xLower_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_xUpper_id,d_FOV_xUpper_id,d_FOV_xUpper_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_yLower_id,d_FOV_yLower_id,d_FOV_yLower_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_yUpper_id,d_FOV_yUpper_id,d_FOV_yUpper_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_zLower_id,d_FOV_zLower_id,d_FOV_zLower_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_FOV_zUpper_id,d_FOV_zUpper_id,d_FOV_zUpper_id, refine_operator_map);
	d_mapping_fill->registerRefine(d_stalled_1_id,d_stalled_1_id,d_stalled_1_id, refine_operator_map);


    d_tagging_fill->registerRefine(d_nonSync_regridding_tag_id,d_nonSync_regridding_tag_id,d_nonSync_regridding_tag_id, d_grid_geometry->lookupRefineOperator(nonSync, "NO_REFINE"));

	//refine and coarsen operators
	string refine_op_name = "LINEAR_REFINE";
	int order = 0;
	if (database->isDatabase("regridding")) {
		regridding_db = database->getDatabase("regridding");
		if (regridding_db->isString("interpolator")) {
			refine_op_name = regridding_db->getString("interpolator");
			if (refine_op_name == "LINEAR_REFINE") {
				order = 1;
			}
			if (refine_op_name == "CUBIC_REFINE") {
				order = 3;
			}
			if (refine_op_name == "QUINTIC_REFINE") {
				order = 5;
			}
		}
	}
	std::shared_ptr< hier::RefineOperator > refine_operator, refine_operator_bound;
	std::shared_ptr< hier::CoarsenOperator > coarsen_operator = d_grid_geometry->lookupCoarsenOperator(FOV_1, "CONSTANT_COARSEN");
	if (order > 0) {
		std::shared_ptr< hier::RefineOperator > tmp_refine_operator(new LagrangianPolynomicRefine(false, order, d_patch_hierarchy, d_dim));
		refine_operator = tmp_refine_operator;
		std::shared_ptr< hier::RefineOperator > tmp_refine_operator_bound(new LagrangianPolynomicRefine(true, order, d_patch_hierarchy, d_dim));
		refine_operator_bound = tmp_refine_operator_bound;
	} else {
		refine_operator = d_grid_geometry->lookupRefineOperator(FOV_1, refine_op_name);
		refine_operator_bound = d_grid_geometry->lookupRefineOperator(FOV_1, refine_op_name);
	}

	std::shared_ptr<SAMRAI::hier::TimeInterpolateOperator> tio_mesh1(new TimeInterpolator(d_grid_geometry, "mesh"));
	time_interpolate_operator_mesh1 = std::dynamic_pointer_cast<TimeInterpolator>(tio_mesh1);


	//Register variables to the refineAlgorithm for boundaries

	if (d_refinedTimeStepping) {
		d_bdry_fill_advance1->registerRefine(d_rk1Psi_id,d_rk1Psi_id,d_Psi_p_id,d_rk1Psi_id,d_rk2Psi_id,d_rk3Psi_id,d_Psi_id,d_rk1Psi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_z_id,d_rk1Phid_z_id,d_Phid_z_p_id,d_rk1Phid_z_id,d_rk2Phid_z_id,d_rk3Phid_z_id,d_Phid_z_id,d_rk1Phid_z_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_y_id,d_rk1Phid_y_id,d_Phid_y_p_id,d_rk1Phid_y_id,d_rk2Phid_y_id,d_rk3Phid_y_id,d_Phid_y_id,d_rk1Phid_y_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_x_id,d_rk1Phid_x_id,d_Phid_x_p_id,d_rk1Phid_x_id,d_rk2Phid_x_id,d_rk3Phid_x_id,d_Phid_x_id,d_rk1Phid_x_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance1->registerRefine(d_rk1phi_id,d_rk1phi_id,d_phi_p_id,d_rk1phi_id,d_rk2phi_id,d_rk3phi_id,d_phi_id,d_rk1phi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance1->registerRefine(d_rk1Psisf_id,d_rk1Psisf_id,d_Psisf_p_id,d_rk1Psisf_id,d_rk2Psisf_id,d_rk3Psisf_id,d_Psisf_id,d_rk1Psisf_id,refine_operator,tio_mesh1);
	} else {
		d_bdry_fill_advance1->registerRefine(d_rk1Psi_id,d_rk1Psi_id,d_rk1Psi_id,refine_operator);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_z_id,d_rk1Phid_z_id,d_rk1Phid_z_id,refine_operator);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_y_id,d_rk1Phid_y_id,d_rk1Phid_y_id,refine_operator);
		d_bdry_fill_advance1->registerRefine(d_rk1Phid_x_id,d_rk1Phid_x_id,d_rk1Phid_x_id,refine_operator);
		d_bdry_fill_advance1->registerRefine(d_rk1phi_id,d_rk1phi_id,d_rk1phi_id,refine_operator);
		d_bdry_fill_advance1->registerRefine(d_rk1Psisf_id,d_rk1Psisf_id,d_rk1Psisf_id,refine_operator);
	}
	if (d_refinedTimeStepping) {
		d_bdry_fill_advance7->registerRefine(d_rk2Psi_id,d_rk2Psi_id,d_Psi_p_id,d_rk1Psi_id,d_rk2Psi_id,d_rk3Psi_id,d_Psi_id,d_rk2Psi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_z_id,d_rk2Phid_z_id,d_Phid_z_p_id,d_rk1Phid_z_id,d_rk2Phid_z_id,d_rk3Phid_z_id,d_Phid_z_id,d_rk2Phid_z_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_y_id,d_rk2Phid_y_id,d_Phid_y_p_id,d_rk1Phid_y_id,d_rk2Phid_y_id,d_rk3Phid_y_id,d_Phid_y_id,d_rk2Phid_y_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_x_id,d_rk2Phid_x_id,d_Phid_x_p_id,d_rk1Phid_x_id,d_rk2Phid_x_id,d_rk3Phid_x_id,d_Phid_x_id,d_rk2Phid_x_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance7->registerRefine(d_rk2phi_id,d_rk2phi_id,d_phi_p_id,d_rk1phi_id,d_rk2phi_id,d_rk3phi_id,d_phi_id,d_rk2phi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance7->registerRefine(d_rk2Psisf_id,d_rk2Psisf_id,d_Psisf_p_id,d_rk1Psisf_id,d_rk2Psisf_id,d_rk3Psisf_id,d_Psisf_id,d_rk2Psisf_id,refine_operator,tio_mesh1);
	} else {
		d_bdry_fill_advance7->registerRefine(d_rk2Psi_id,d_rk2Psi_id,d_rk2Psi_id,refine_operator);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_z_id,d_rk2Phid_z_id,d_rk2Phid_z_id,refine_operator);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_y_id,d_rk2Phid_y_id,d_rk2Phid_y_id,refine_operator);
		d_bdry_fill_advance7->registerRefine(d_rk2Phid_x_id,d_rk2Phid_x_id,d_rk2Phid_x_id,refine_operator);
		d_bdry_fill_advance7->registerRefine(d_rk2phi_id,d_rk2phi_id,d_rk2phi_id,refine_operator);
		d_bdry_fill_advance7->registerRefine(d_rk2Psisf_id,d_rk2Psisf_id,d_rk2Psisf_id,refine_operator);
	}
	if (d_refinedTimeStepping) {
		d_bdry_fill_advance13->registerRefine(d_rk3Psi_id,d_rk3Psi_id,d_Psi_p_id,d_rk1Psi_id,d_rk2Psi_id,d_rk3Psi_id,d_Psi_id,d_rk3Psi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_z_id,d_rk3Phid_z_id,d_Phid_z_p_id,d_rk1Phid_z_id,d_rk2Phid_z_id,d_rk3Phid_z_id,d_Phid_z_id,d_rk3Phid_z_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_y_id,d_rk3Phid_y_id,d_Phid_y_p_id,d_rk1Phid_y_id,d_rk2Phid_y_id,d_rk3Phid_y_id,d_Phid_y_id,d_rk3Phid_y_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_x_id,d_rk3Phid_x_id,d_Phid_x_p_id,d_rk1Phid_x_id,d_rk2Phid_x_id,d_rk3Phid_x_id,d_Phid_x_id,d_rk3Phid_x_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance13->registerRefine(d_rk3phi_id,d_rk3phi_id,d_phi_p_id,d_rk1phi_id,d_rk2phi_id,d_rk3phi_id,d_phi_id,d_rk3phi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance13->registerRefine(d_rk3Psisf_id,d_rk3Psisf_id,d_Psisf_p_id,d_rk1Psisf_id,d_rk2Psisf_id,d_rk3Psisf_id,d_Psisf_id,d_rk3Psisf_id,refine_operator,tio_mesh1);
	} else {
		d_bdry_fill_advance13->registerRefine(d_rk3Psi_id,d_rk3Psi_id,d_rk3Psi_id,refine_operator);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_z_id,d_rk3Phid_z_id,d_rk3Phid_z_id,refine_operator);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_y_id,d_rk3Phid_y_id,d_rk3Phid_y_id,refine_operator);
		d_bdry_fill_advance13->registerRefine(d_rk3Phid_x_id,d_rk3Phid_x_id,d_rk3Phid_x_id,refine_operator);
		d_bdry_fill_advance13->registerRefine(d_rk3phi_id,d_rk3phi_id,d_rk3phi_id,refine_operator);
		d_bdry_fill_advance13->registerRefine(d_rk3Psisf_id,d_rk3Psisf_id,d_rk3Psisf_id,refine_operator);
	}
	if (d_refinedTimeStepping) {
		d_bdry_fill_advance19->registerRefine(d_Psi_id,d_Psi_id,d_Psi_p_id,d_rk1Psi_id,d_rk2Psi_id,d_rk3Psi_id,d_Psi_id,d_Psi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance19->registerRefine(d_Phid_z_id,d_Phid_z_id,d_Phid_z_p_id,d_rk1Phid_z_id,d_rk2Phid_z_id,d_rk3Phid_z_id,d_Phid_z_id,d_Phid_z_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance19->registerRefine(d_Phid_y_id,d_Phid_y_id,d_Phid_y_p_id,d_rk1Phid_y_id,d_rk2Phid_y_id,d_rk3Phid_y_id,d_Phid_y_id,d_Phid_y_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance19->registerRefine(d_Phid_x_id,d_Phid_x_id,d_Phid_x_p_id,d_rk1Phid_x_id,d_rk2Phid_x_id,d_rk3Phid_x_id,d_Phid_x_id,d_Phid_x_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance19->registerRefine(d_phi_id,d_phi_id,d_phi_p_id,d_rk1phi_id,d_rk2phi_id,d_rk3phi_id,d_phi_id,d_phi_id,refine_operator,tio_mesh1);
		d_bdry_fill_advance19->registerRefine(d_Psisf_id,d_Psisf_id,d_Psisf_p_id,d_rk1Psisf_id,d_rk2Psisf_id,d_rk3Psisf_id,d_Psisf_id,d_Psisf_id,refine_operator,tio_mesh1);
	} else {
		d_bdry_fill_advance19->registerRefine(d_Psi_id,d_Psi_id,d_Psi_id,refine_operator);
		d_bdry_fill_advance19->registerRefine(d_Phid_z_id,d_Phid_z_id,d_Phid_z_id,refine_operator);
		d_bdry_fill_advance19->registerRefine(d_Phid_y_id,d_Phid_y_id,d_Phid_y_id,refine_operator);
		d_bdry_fill_advance19->registerRefine(d_Phid_x_id,d_Phid_x_id,d_Phid_x_id,refine_operator);
		d_bdry_fill_advance19->registerRefine(d_phi_id,d_phi_id,d_phi_id,refine_operator);
		d_bdry_fill_advance19->registerRefine(d_Psisf_id,d_Psisf_id,d_Psisf_id,refine_operator);
	}
	d_bdry_fill_analysis1->registerRefine(d_Chi_norm_id,d_Chi_norm_id,d_Chi_norm_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_Chi_z_id,d_Chi_z_id,d_Chi_z_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_Chi_y_id,d_Chi_y_id,d_Chi_y_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_Chi_x_id,d_Chi_x_id,d_Chi_x_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_Chi_t_id,d_Chi_t_id,d_Chi_t_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_d2tphi_id,d_d2tphi_id,d_d2tphi_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vorticity_norm_id,d_vorticity_norm_id,d_vorticity_norm_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vzminus_id,d_vzminus_id,d_vzminus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vzplus_id,d_vzplus_id,d_vzplus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vyminus_id,d_vyminus_id,d_vyminus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vyplus_id,d_vyplus_id,d_vyplus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vxminus_id,d_vxminus_id,d_vxminus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_vxplus_id,d_vxplus_id,d_vxplus_id,refine_operator);
	d_bdry_fill_analysis1->registerRefine(d_phi22_id,d_phi22_id,d_phi22_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_Psisf_id,d_Psisf_id,d_Psisf_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_Psisf_id,d_Psisf_id,d_Psisf_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_phi_id,d_phi_id,d_phi_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_phi_id,d_phi_id,d_phi_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_Phid_x_id,d_Phid_x_id,d_Phid_x_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_Phid_x_id,d_Phid_x_id,d_Phid_x_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_Phid_y_id,d_Phid_y_id,d_Phid_y_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_Phid_y_id,d_Phid_y_id,d_Phid_y_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_Phid_z_id,d_Phid_z_id,d_Phid_z_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_Phid_z_id,d_Phid_z_id,d_Phid_z_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_Psi_id,d_Psi_id,d_Psi_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_Psi_id,d_Psi_id,d_Psi_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_phi2_id,d_phi2_id,d_phi2_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_phi2_id,d_phi2_id,d_phi2_id,refine_operator);
	d_bdry_fill_init->registerRefine(d_dphi2_id,d_dphi2_id,d_dphi2_id,refine_operator);
	d_bdry_post_coarsen->registerRefine(d_dphi2_id,d_dphi2_id,d_dphi2_id,refine_operator);


	//Register variables to the refineAlgorithm for filling new levels on regridding
	d_fill_new_level->registerRefine(d_Psisf_id,d_Psisf_id,d_Psisf_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_phi_id,d_phi_id,d_phi_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Phid_x_id,d_Phid_x_id,d_Phid_x_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Phid_y_id,d_Phid_y_id,d_Phid_y_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Phid_z_id,d_Phid_z_id,d_Phid_z_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Psi_id,d_Psi_id,d_Psi_id,refine_operator_bound);
	d_fill_new_level_aux->registerRefine(d_phi2_id,d_phi2_id,d_phi2_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_phi2_id,d_phi2_id,d_phi2_id,refine_operator_bound);
	d_fill_new_level_aux->registerRefine(d_dphi2_id,d_dphi2_id,d_dphi2_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_dphi2_id,d_dphi2_id,d_dphi2_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_phi22_id,d_phi22_id,d_phi22_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vxplus_id,d_vxplus_id,d_vxplus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vxminus_id,d_vxminus_id,d_vxminus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vyplus_id,d_vyplus_id,d_vyplus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vyminus_id,d_vyminus_id,d_vyminus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vzplus_id,d_vzplus_id,d_vzplus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vzminus_id,d_vzminus_id,d_vzminus_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_vorticity_norm_id,d_vorticity_norm_id,d_vorticity_norm_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_d2tphi_id,d_d2tphi_id,d_d2tphi_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Chi_t_id,d_Chi_t_id,d_Chi_t_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Chi_x_id,d_Chi_x_id,d_Chi_x_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Chi_y_id,d_Chi_y_id,d_Chi_y_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Chi_z_id,d_Chi_z_id,d_Chi_z_id,refine_operator_bound);
	d_fill_new_level->registerRefine(d_Chi_norm_id,d_Chi_norm_id,d_Chi_norm_id,refine_operator_bound);


	//Register variables to the coarsenAlgorithm
	d_coarsen_algorithm->registerCoarsen(d_Psisf_id,d_Psisf_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_phi_id,d_phi_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_Phid_x_id,d_Phid_x_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_Phid_y_id,d_Phid_y_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_Phid_z_id,d_Phid_z_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_Psi_id,d_Psi_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_phi2_id,d_phi2_id,coarsen_operator);
	d_coarsen_algorithm->registerCoarsen(d_dphi2_id,d_dphi2_id,coarsen_operator);

	readTable("drphir1d.h5", &coord0_data_drphir1d, coord0_size_drphir1d, coord0_dx_drphir1d, &vars_data_drphir1d, vars_size_drphir1d);
	readTable("phir1d.h5", &coord0_data_phir1d, coord0_size_phir1d, coord0_dx_phir1d, &vars_data_phir1d, vars_size_phir1d);


    Commons::initialization();
}

/*
 * Destructor.
 */
Problem::~Problem() 
{
} 

/*
 * Initialize the data from a given level.
 */
void Problem::initializeLevelData (
   const std::shared_ptr<hier::PatchHierarchy >& hierarchy , 
   const int level_number ,
   const double init_data_time ,
   const bool can_be_refined ,
   const bool initial_time ,
   const std::shared_ptr<hier::PatchLevel >& old_level ,
   const bool allocate_data )
{
    cout<<"Initializing level "<<level_number<<endl;
    tbox::MemoryUtilities::printMemoryInfo(cout);   
	std::shared_ptr< hier::PatchLevel > level(hierarchy->getPatchLevel(level_number));

	// Allocate storage needed to initialize level and fill data from coarser levels in AMR hierarchy.  
	level->allocatePatchData(d_interior_regridding_value_id, init_data_time);
	level->allocatePatchData(d_nonSync_regridding_tag_id, init_data_time);
	level->allocatePatchData(d_interior_i_id, init_data_time);
	level->allocatePatchData(d_interior_j_id, init_data_time);
	level->allocatePatchData(d_interior_k_id, init_data_time);
	level->allocatePatchData(d_FOV_1_id, init_data_time);
	level->allocatePatchData(d_FOV_xLower_id, init_data_time);
	level->allocatePatchData(d_FOV_xUpper_id, init_data_time);
	level->allocatePatchData(d_FOV_yLower_id, init_data_time);
	level->allocatePatchData(d_FOV_yUpper_id, init_data_time);
	level->allocatePatchData(d_FOV_zLower_id, init_data_time);
	level->allocatePatchData(d_FOV_zUpper_id, init_data_time);
	level->allocatePatchData(d_Psisf_id, init_data_time);
	level->allocatePatchData(d_phi_id, init_data_time);
	level->allocatePatchData(d_Phid_x_id, init_data_time);
	level->allocatePatchData(d_Phid_y_id, init_data_time);
	level->allocatePatchData(d_Phid_z_id, init_data_time);
	level->allocatePatchData(d_Psi_id, init_data_time);
	level->allocatePatchData(d_phi2_id, init_data_time);
	level->allocatePatchData(d_dphi2_id, init_data_time);
	level->allocatePatchData(d_phi22_id, init_data_time);
	level->allocatePatchData(d_vxplus_id, init_data_time);
	level->allocatePatchData(d_vxminus_id, init_data_time);
	level->allocatePatchData(d_vyplus_id, init_data_time);
	level->allocatePatchData(d_vyminus_id, init_data_time);
	level->allocatePatchData(d_vzplus_id, init_data_time);
	level->allocatePatchData(d_vzminus_id, init_data_time);
	level->allocatePatchData(d_vorticity_norm_id, init_data_time);
	level->allocatePatchData(d_d2tphi_id, init_data_time);
	level->allocatePatchData(d_Chi_t_id, init_data_time);
	level->allocatePatchData(d_Chi_x_id, init_data_time);
	level->allocatePatchData(d_Chi_y_id, init_data_time);
	level->allocatePatchData(d_Chi_z_id, init_data_time);
	level->allocatePatchData(d_Chi_norm_id, init_data_time);
	level->allocatePatchData(d_rk1Psisf_id, init_data_time);
	level->allocatePatchData(d_rk1phi_id, init_data_time);
	level->allocatePatchData(d_rk1Phid_x_id, init_data_time);
	level->allocatePatchData(d_rk1Phid_y_id, init_data_time);
	level->allocatePatchData(d_rk1Phid_z_id, init_data_time);
	level->allocatePatchData(d_rk1Psi_id, init_data_time);
	level->allocatePatchData(d_rk2Psisf_id, init_data_time);
	level->allocatePatchData(d_rk2phi_id, init_data_time);
	level->allocatePatchData(d_rk2Phid_x_id, init_data_time);
	level->allocatePatchData(d_rk2Phid_y_id, init_data_time);
	level->allocatePatchData(d_rk2Phid_z_id, init_data_time);
	level->allocatePatchData(d_rk2Psi_id, init_data_time);
	level->allocatePatchData(d_rk3Psisf_id, init_data_time);
	level->allocatePatchData(d_rk3phi_id, init_data_time);
	level->allocatePatchData(d_rk3Phid_x_id, init_data_time);
	level->allocatePatchData(d_rk3Phid_y_id, init_data_time);
	level->allocatePatchData(d_rk3Phid_z_id, init_data_time);
	level->allocatePatchData(d_rk3Psi_id, init_data_time);
	level->allocatePatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id, init_data_time);
	level->allocatePatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id, init_data_time);
	level->allocatePatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id, init_data_time);
	level->allocatePatchData(d_stalled_1_id, init_data_time);
	level->allocatePatchData(d_Psisf_p_id, init_data_time);
	level->allocatePatchData(d_phi_p_id, init_data_time);
	level->allocatePatchData(d_Phid_x_p_id, init_data_time);
	level->allocatePatchData(d_Phid_y_p_id, init_data_time);
	level->allocatePatchData(d_Phid_z_p_id, init_data_time);
	level->allocatePatchData(d_Psi_p_id, init_data_time);
	level->allocatePatchData(d_mask_id, init_data_time);


	//Mapping the current data for new level.
	if (initial_time || level_number == 0) {
		mapDataOnPatch(init_data_time, initial_time, level_number, level);
	}

	//Fill a finer level with the data of the next coarse level.
	if ((level_number > 0) || old_level) {
		d_mapping_fill->createSchedule(level,old_level,level_number-1,hierarchy,this)->fillData(init_data_time, false);
		correctFOVS(level);
	}

	//Fill a finer level with the data of the next coarse level.
	if (!initial_time && ((level_number > 0) || old_level)) {
		d_fill_new_level->createSchedule(level,old_level,level_number-1,hierarchy,this)->fillData(init_data_time, false);
		postNewLevel(level);
		d_fill_new_level_aux->createSchedule(level,this)->fillData(init_data_time, false);
	}

	//Interphase mapping
	if (initial_time || level_number == 0) {
		interphaseMapping(init_data_time, initial_time, level_number, level, 1);
	}


	//Initialize current data for new level.
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr< hier::Patch > patch = *p_it;
		if (initial_time) {
  		    initializeDataOnPatch(*patch, init_data_time, initial_time);
		}

	}
	//Post-initialization Sync.
    	if (initial_time || level_number == 0) {

		//First synchronization from initialization
		d_bdry_fill_init->createSchedule(level,this)->fillData(init_data_time, false);
		double current_time = init_data_time;
		const double level_ratio = level->getRatioToCoarserLevel().max();
		double simPlat_dt = 0;
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr< hier::Patch >& patch = *p_it;
			double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
			double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
			double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
			double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
			double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
			double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
			double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
			double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
			double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
			double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
			double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
			double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
			double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();

			//Get the dimensions of the patch
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast = patch->getBox().upper();

			//Get delta spaces into an array. dx, dy, dz.
			const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
			const double* dx  = patch_geom->getDx();
			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
			for (int k = 0; k < klast; k++) {
				for (int j = 0; j < jlast; j++) {
					for (int i = 0; i < ilast; i++) {
						if (((i + 3 < ilast || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) && (i - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) && (j + 3 < jlast || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) && (j - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) && (k + 3 < klast || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) && (k - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)))) {
							vector(phi2, i, j, k) = vector(phi, i, j, k) * vector(phi, i, j, k);
							vector(dphi2, i, j, k) = vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k) + vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k) + vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k);
						}
					}
				}
			}
		}
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr< hier::Patch >& patch = *p_it;
			double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
			double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
			double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
			double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
			double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
			double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
			double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
			double* Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_id).get())->getPointer();
			double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
			double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
			double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
			double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
			double* Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_id).get())->getPointer();
			double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
			double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();

			//Hard region field distance variables
			double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();

			//Get the dimensions of the patch
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast = patch->getBox().upper();

			//Get delta spaces into an array. dx, dy, dz.
			const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
			const double* dx  = patch_geom->getDx();
			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
			for (int k = 0; k < klast; k++) {
				for (int j = 0; j < jlast; j++) {
					for (int i = 0; i < ilast; i++) {
						if ((vector(FOV_xLower, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
						if ((vector(FOV_xUpper, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
						if ((vector(FOV_yLower, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
						if ((vector(FOV_yUpper, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
						if ((vector(FOV_zLower, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
						if ((vector(FOV_zUpper, i, j, k) > 0)) {
							//Region field extrapolations
							if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
								extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
							}
						}
					}
				}
			}
		}
		//Last synchronization from initialization
		d_bdry_fill_init->createSchedule(level,this)->fillData(init_data_time, false);

    	}

    cout<<"Level "<<level_number<<" initialized"<<endl;
    tbox::MemoryUtilities::printMemoryInfo(cout);
}

void Problem::postInit(){
	double current_time = 0;
	for (int ln=0; ln<=d_patch_hierarchy->getFinestLevelNumber(); ++ln ) {
		std::shared_ptr<hier::PatchLevel > level(d_patch_hierarchy->getPatchLevel(ln));
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr<hier::Patch >& patch = *p_it;
	
			double* Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_id).get())->getPointer();
			double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
			double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
			double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
			double* d2tphi = ((pdat::NodeData<double> *) patch->getPatchData(d_d2tphi_id).get())->getPointer();
			double* Chi_t = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_t_id).get())->getPointer();
			double* Chi_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_x_id).get())->getPointer();
			double* Chi_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_y_id).get())->getPointer();
			double* Chi_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_z_id).get())->getPointer();
			double* phi22 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi22_id).get())->getPointer();
			double* vxplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vxplus_id).get())->getPointer();
			double* vxminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vxminus_id).get())->getPointer();
			double* vyplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vyplus_id).get())->getPointer();
			double* vyminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vyminus_id).get())->getPointer();
			double* vzplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vzplus_id).get())->getPointer();
			double* vzminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vzminus_id).get())->getPointer();
			double* vorticity_norm = ((pdat::NodeData<double> *) patch->getPatchData(d_vorticity_norm_id).get())->getPointer();
			double* Chi_norm = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_norm_id).get())->getPointer();
			double d_Vtx_o0_t2_m0_l0, d_Vtx_o0_t1_m0_l0, d_Vtx_o0_t0_m0_l0, d_Vtx_o0_t6_m0_l0, d_Vtx_o0_t5_m0_l0, d_Vtx_o0_t4_m0_l0, d_phi22_o0_t7_m0_l0, d_Vty_o0_t6_m0_l0, d_Vty_o0_t5_m0_l0, d_phi22_o0_t10_m0_l0, d_phi22_o0_t11_m0_l0, d_Vtz_o0_t6_m0_l0, d_Vtx_o0_t7_m0_l0, m_Vtx_o0_t0_l0, m_Vtx_o0_t1_l0, m_Vtx_o0_t2_l0, m_Vtx_o0_t3_l0, m_Vtx_o0_t4_l0, m_Vtx_o0_t5_l0, m_Vtx_o0_t6_l0, Vtx, d_Vty_o0_t7_m0_l0, m_Vty_o0_t0_l0, m_Vty_o0_t1_l0, m_Vty_o0_t2_l0, m_Vty_o0_t3_l0, m_Vty_o0_t4_l0, m_Vty_o0_t5_l0, m_Vty_o0_t6_l0, Vty, d_Vtz_o0_t7_m0_l0, m_Vtz_o0_t0_l0, m_Vtz_o0_t1_l0, m_Vtz_o0_t2_l0, m_Vtz_o0_t3_l0, m_Vtz_o0_t4_l0, m_Vtz_o0_t5_l0, m_Vtz_o0_t6_l0, Vtz, m_Vxz_o0_t0_l0, m_Vxz_o0_t1_l0, m_Vxz_o0_t2_l0, m_Vxz_o0_t3_l0, m_Vxz_o0_t4_l0, m_Vxz_o0_t5_l0, m_Vxz_o0_t6_l0, m_Vxz_o0_t7_l0, Vxz, m_Vyz_o0_t0_l0, m_Vyz_o0_t1_l0, m_Vyz_o0_t2_l0, m_Vyz_o0_t3_l0, m_Vyz_o0_t4_l0, m_Vyz_o0_t5_l0, m_Vyz_o0_t6_l0, m_Vyz_o0_t7_l0, Vyz, m_Vxy_o0_t0_l0, m_Vxy_o0_t1_l0, m_Vxy_o0_t2_l0, m_Vxy_o0_t3_l0, m_Vxy_o0_t4_l0, m_Vxy_o0_t5_l0, m_Vxy_o0_t6_l0, m_Vxy_o0_t7_l0, Vxy, Phiu_x, Phiu_z, Phiu_y, Xphi, KXXf, KXf, r1_z, r2_z, rorbit_dynamic, omegaorbit, r2_y, r2_x, sq_r2, r1_y, r1_x, sq_r1, Tbar, d_phi22_o0_t0_m0_l0, d_vxplus_o0_t0_m0_l0, d_vxminus_o0_t0_m0_l0, d_vyplus_o0_t0_m0_l0, d_vyminus_o0_t0_m0_l0, d_vzplus_o0_t0_m0_l0, d_vzminus_o0_t0_m0_l0, d_vorticity_norm_o0_t0_m0_l0, d_d2tphi_o0_t12_m0_l0, d_Chi_t_o0_t0_m0_l0, d_Chi_x_o0_t0_m0_l0, d_Chi_y_o0_t0_m0_l0, d_Chi_z_o0_t0_m0_l0, d_Chi_norm_o0_t0_m0_l0, m_phi22_o0_t1_l0, m_phi22_o0_t2_l0, m_phi22_o0_t3_l0, m_phi22_o0_t4_l0, m_phi22_o0_t5_l0, m_phi22_o0_t6_l0, m_phi22_o0_t7_l0, m_phi22_o0_t8_l0, m_phi22_o0_t9_l0, m_phi22_o0_t10_l0, m_phi22_o0_t11_l0, m_phi22_o0_t12_l0, m_d2tphi_o0_t0_l0, m_d2tphi_o0_t1_l0, m_d2tphi_o0_t2_l0, m_d2tphi_o0_t3_l0, m_d2tphi_o0_t4_l0, m_d2tphi_o0_t5_l0, m_d2tphi_o0_t6_l0, m_d2tphi_o0_t7_l0, m_d2tphi_o0_t8_l0, m_d2tphi_o0_t9_l0, m_d2tphi_o0_t10_l0, m_d2tphi_o0_t11_l0;
	
			//Get the dimensions of the patch
			hier::Box pbox = patch->getBox();
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast = patch->getBox().upper();
	
			//Get delta spaces into an array. dx, dy, dz.
			std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
			const double* dx  = patch_geom->getDx();
	
			//Auxiliary definitions
			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
						if ((i + 2 < ilast && i - 2 >= 0 && j + 2 < jlast && j - 2 >= 0 && k + 2 < klast && k - 2 >= 0)) {
							d_Vtx_o0_t2_m0_l0 = D1CDO4_i(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t1_m0_l0 = D1CDO4_j(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t0_m0_l0 = D1CDO4_k(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t6_m0_l0 = D1CDO4_i(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t5_m0_l0 = D1CDO4_j(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t4_m0_l0 = D1CDO4_k(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_phi22_o0_t7_m0_l0 = D1CDO4_i(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vty_o0_t6_m0_l0 = D1CDO4_j(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vty_o0_t5_m0_l0 = D1CDO4_k(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_phi22_o0_t10_m0_l0 = D1CDO4_i(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_phi22_o0_t11_m0_l0 = D1CDO4_j(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtz_o0_t6_m0_l0 = D1CDO4_k(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							d_Vtx_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_x, i, j, k) * vector(Psisf, i, j, k);
							m_Vtx_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t0_m0_l0;
							m_Vtx_o0_t1_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t1_m0_l0;
							m_Vtx_o0_t2_l0 = (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t2_m0_l0;
							m_Vtx_o0_t3_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
							m_Vtx_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_Vtx_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t5_m0_l0;
							m_Vtx_o0_t6_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t6_m0_l0;
							Vtx = ((((((m_Vtx_o0_t0_l0 + m_Vtx_o0_t1_l0) + m_Vtx_o0_t2_l0) + m_Vtx_o0_t3_l0) + m_Vtx_o0_t4_l0) + m_Vtx_o0_t5_l0) + m_Vtx_o0_t6_l0) + d_Vtx_o0_t7_m0_l0;
							d_Vty_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_y, i, j, k) * vector(Psisf, i, j, k);
							m_Vty_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t0_m0_l0;
							m_Vty_o0_t1_l0 = (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtx_o0_t1_m0_l0;
							m_Vty_o0_t2_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t1_m0_l0;
							m_Vty_o0_t3_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t2_m0_l0;
							m_Vty_o0_t4_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t5_m0_l0;
							m_Vty_o0_t5_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vty_o0_t6_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t6_m0_l0;
							Vty = ((((((m_Vty_o0_t0_l0 + m_Vty_o0_t1_l0) + m_Vty_o0_t2_l0) + m_Vty_o0_t3_l0) + m_Vty_o0_t4_l0) + m_Vty_o0_t5_l0) + m_Vty_o0_t6_l0) + d_Vty_o0_t7_m0_l0;
							d_Vtz_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_z, i, j, k) * vector(Psisf, i, j, k);
							m_Vtz_o0_t0_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vtx_o0_t0_m0_l0;
							m_Vtz_o0_t1_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t0_m0_l0;
							m_Vtz_o0_t2_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t1_m0_l0;
							m_Vtz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t2_m0_l0;
							m_Vtz_o0_t4_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_Vtz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vtz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtz_o0_t6_m0_l0;
							Vtz = ((((((m_Vtz_o0_t0_l0 + m_Vtz_o0_t1_l0) + m_Vtz_o0_t2_l0) + m_Vtz_o0_t3_l0) + m_Vtz_o0_t4_l0) + m_Vtz_o0_t5_l0) + m_Vtz_o0_t6_l0) + d_Vtz_o0_t7_m0_l0;
							m_Vxz_o0_t0_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_Vxz_o0_t1_l0 = (-vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_Vxz_o0_t2_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t5_m0_l0;
							m_Vxz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t6_m0_l0;
							m_Vxz_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
							m_Vxz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vxz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtz_o0_t6_m0_l0;
							m_Vxz_o0_t7_l0 = vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
							Vxz = ((((((m_Vxz_o0_t0_l0 + m_Vxz_o0_t1_l0) + m_Vxz_o0_t2_l0) + m_Vxz_o0_t3_l0) + m_Vxz_o0_t4_l0) + m_Vxz_o0_t5_l0) + m_Vxz_o0_t6_l0) + m_Vxz_o0_t7_l0;
							m_Vyz_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t5_m0_l0;
							m_Vyz_o0_t1_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vyz_o0_t2_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vyz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vty_o0_t6_m0_l0;
							m_Vyz_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t1_m0_l0;
							m_Vyz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_Vyz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtz_o0_t6_m0_l0;
							m_Vyz_o0_t7_l0 = vector(Phid_y, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
							Vyz = ((((((m_Vyz_o0_t0_l0 + m_Vyz_o0_t1_l0) + m_Vyz_o0_t2_l0) + m_Vyz_o0_t3_l0) + m_Vyz_o0_t4_l0) + m_Vyz_o0_t5_l0) + m_Vyz_o0_t6_l0) + m_Vyz_o0_t7_l0;
							m_Vxy_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t4_m0_l0;
							m_Vxy_o0_t1_l0 = (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtx_o0_t5_m0_l0;
							m_Vxy_o0_t2_l0 = (-vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t5_m0_l0;
							m_Vxy_o0_t3_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t6_m0_l0;
							m_Vxy_o0_t4_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
							m_Vxy_o0_t5_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_Vxy_o0_t6_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t6_m0_l0;
							m_Vxy_o0_t7_l0 = vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t1_m0_l0;
							Vxy = ((((((m_Vxy_o0_t0_l0 + m_Vxy_o0_t1_l0) + m_Vxy_o0_t2_l0) + m_Vxy_o0_t3_l0) + m_Vxy_o0_t4_l0) + m_Vxy_o0_t5_l0) + m_Vxy_o0_t6_l0) + m_Vxy_o0_t7_l0;
							Phiu_x = vector(Phid_x, i, j, k);
							Phiu_z = vector(Phid_z, i, j, k);
							Phiu_y = vector(Phid_y, i, j, k);
							Xphi = (-vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) + Phiu_x * vector(Phid_x, i, j, k) + Phiu_y * vector(Phid_y, i, j, k) + Phiu_z * vector(Phid_z, i, j, k);
							KXXf = 0.5 * betax + gammax * Xphi;
							KXf = (-0.5 * sigmax) + 0.5 * betax * Xphi + 0.5 * gammax * (Xphi * Xphi);
							r1_z = zcoord(k);
							r2_z = zcoord(k);
							rorbit_dynamic = MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, ((current_time - tini) - tslow) - torbit)))) * rorbit + MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, ((current_time - tini) - tslow) - torbit)))) * MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, (current_time - tini) - torbit)))) * (rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0)) + rstart * MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, (current_time - tini) - torbit))));
							omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rorbit_dynamic) * (2.0 * rorbit_dynamic) * (2.0 * rorbit_dynamic))) * (MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, current_time - torbit)))) * MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, current_time)))) * 4.0 * (2.0 * torbit - current_time) * current_time / ((2.0 * torbit) * (2.0 * torbit)) + MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, current_time - torbit)))));
							r2_y = ycoord(j) - 2.0 / (1.0 + mu) * rorbit_dynamic * sin(omegaorbit * current_time);
							r2_x = xcoord(i) - 2.0 / (1.0 + mu) * rorbit_dynamic * cos(omegaorbit * current_time);
							sq_r2 = r2_x * r2_x + r2_y * r2_y + r2_z * r2_z;
							r1_y = ycoord(j) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic * sin(omegaorbit * current_time);
							r1_x = xcoord(i) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic * cos(omegaorbit * current_time);
							sq_r1 = r1_x * r1_x + r1_y * r1_y + r1_z * r1_z;
							Tbar = -(sq_r1 * exp(-((sqrt(sq_r1) - rdonut) * (sqrt(sq_r1) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * sq_r2 * exp(-((sqrt(sq_r2) - rdonut) * (sqrt(sq_r2) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
							d_phi22_o0_t0_m0_l0 = -1.0 / (4.0 * Mpl_cte) * Tbar / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)));
							d_vxplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vxminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vyplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vyminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vzplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vzminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
							d_vorticity_norm_o0_t0_m0_l0 = ((-2.0) * (Vtx * Vtx) - 2.0 * (Vty * Vty)) - 2.0 * (Vtz * Vtz) + 2.0 * (Vxy * Vxy) + 2.0 * (Vxz * Vxz) + 2.0 * (Vyz * Vyz);
							d_d2tphi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * Tbar / 4.0;
							d_Chi_t_o0_t0_m0_l0 = KXf * vector(Psisf, i, j, k);
							d_Chi_x_o0_t0_m0_l0 = KXf * vector(Phid_x, i, j, k);
							d_Chi_y_o0_t0_m0_l0 = KXf * vector(Phid_y, i, j, k);
							d_Chi_z_o0_t0_m0_l0 = KXf * vector(Phid_z, i, j, k);
							d_Chi_norm_o0_t0_m0_l0 = (-vector(Chi_t, i, j, k) * vector(Chi_t, i, j, k)) + vector(Chi_x, i, j, k) * vector(Chi_x, i, j, k) + vector(Chi_y, i, j, k) * vector(Chi_y, i, j, k) + vector(Chi_z, i, j, k) * vector(Chi_z, i, j, k);
							m_phi22_o0_t1_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t6_m0_l0;
							m_phi22_o0_t2_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vty_o0_t6_m0_l0;
							m_phi22_o0_t3_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtz_o0_t6_m0_l0;
							m_phi22_o0_t4_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t2_m0_l0;
							m_phi22_o0_t5_l0 = (2.0 * KXXf * (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t6_m0_l0;
							m_phi22_o0_t6_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t1_m0_l0;
							m_phi22_o0_t7_l0 = (4.0 * KXXf * vector(Phid_x, i, j, k) * vector(Phid_y, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t7_m0_l0;
							m_phi22_o0_t8_l0 = (2.0 * KXXf * (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vty_o0_t6_m0_l0;
							m_phi22_o0_t9_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t0_m0_l0;
							m_phi22_o0_t10_l0 = (4.0 * KXXf * vector(Phid_x, i, j, k) * vector(Phid_z, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t10_m0_l0;
							m_phi22_o0_t11_l0 = (4.0 * KXXf * vector(Phid_y, i, j, k) * vector(Phid_z, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t11_m0_l0;
							m_phi22_o0_t12_l0 = (2.0 * KXXf * (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtz_o0_t6_m0_l0;
							m_d2tphi_o0_t0_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k))) * d_Vtx_o0_t6_m0_l0;
							m_d2tphi_o0_t1_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t5_m0_l0;
							m_d2tphi_o0_t2_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
							m_d2tphi_o0_t3_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t2_m0_l0;
							m_d2tphi_o0_t4_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k))) * d_Vty_o0_t6_m0_l0;
							m_d2tphi_o0_t5_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k)) * d_Vty_o0_t5_m0_l0;
							m_d2tphi_o0_t6_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_y, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t1_m0_l0;
							m_d2tphi_o0_t7_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k))) * d_Vtz_o0_t6_m0_l0;
							m_d2tphi_o0_t8_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
							m_d2tphi_o0_t9_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vtz_o0_t6_m0_l0;
							m_d2tphi_o0_t10_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vty_o0_t6_m0_l0;
							m_d2tphi_o0_t11_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vtx_o0_t6_m0_l0;
							vector(phi22, i, j, k) = (((((((((((d_phi22_o0_t0_m0_l0 + m_phi22_o0_t1_l0) + m_phi22_o0_t2_l0) + m_phi22_o0_t3_l0) + m_phi22_o0_t4_l0) + m_phi22_o0_t5_l0) + m_phi22_o0_t6_l0) + m_phi22_o0_t7_l0) + m_phi22_o0_t8_l0) + m_phi22_o0_t9_l0) + m_phi22_o0_t10_l0) + m_phi22_o0_t11_l0) + m_phi22_o0_t12_l0;
							vector(vxplus, i, j, k) = d_vxplus_o0_t0_m0_l0;
							vector(vxminus, i, j, k) = d_vxminus_o0_t0_m0_l0;
							vector(vyplus, i, j, k) = d_vyplus_o0_t0_m0_l0;
							vector(vyminus, i, j, k) = d_vyminus_o0_t0_m0_l0;
							vector(vzplus, i, j, k) = d_vzplus_o0_t0_m0_l0;
							vector(vzminus, i, j, k) = d_vzminus_o0_t0_m0_l0;
							vector(vorticity_norm, i, j, k) = d_vorticity_norm_o0_t0_m0_l0;
							vector(d2tphi, i, j, k) = (((((((((((m_d2tphi_o0_t0_l0 + m_d2tphi_o0_t1_l0) + m_d2tphi_o0_t2_l0) + m_d2tphi_o0_t3_l0) + m_d2tphi_o0_t4_l0) + m_d2tphi_o0_t5_l0) + m_d2tphi_o0_t6_l0) + m_d2tphi_o0_t7_l0) + m_d2tphi_o0_t8_l0) + m_d2tphi_o0_t9_l0) + m_d2tphi_o0_t10_l0) + m_d2tphi_o0_t11_l0) + d_d2tphi_o0_t12_m0_l0;
							vector(Chi_t, i, j, k) = d_Chi_t_o0_t0_m0_l0;
							vector(Chi_x, i, j, k) = d_Chi_x_o0_t0_m0_l0;
							vector(Chi_y, i, j, k) = d_Chi_y_o0_t0_m0_l0;
							vector(Chi_z, i, j, k) = d_Chi_z_o0_t0_m0_l0;
							vector(Chi_norm, i, j, k) = d_Chi_norm_o0_t0_m0_l0;
						}
					}
				}
			}
		}
	}
	
}


void Problem::initializeLevelIntegrator(
   const std::shared_ptr<mesh::GriddingAlgorithmStrategy>& gridding_alg)
{
}

double Problem::getLevelDt(
   const std::shared_ptr<hier::PatchLevel>& level,
   const double dt_time,
   const bool initial_time)
{
  
   TBOX_ASSERT(level);

   if (level->getLevelNumber() == 0) return initial_dt;

    double dt = initial_dt;
    const hier::IntVector ratio = level->getRatioToLevelZero();
    double local_dt = dt;
    for (int i = 0; i < 2; i++) {
        if (local_dt > dt / ratio[i]) {
            local_dt = dt / ratio[i];
        }
    }
    return local_dt;
}

double Problem::getMaxFinerLevelDt(
   const int finer_level_number,
   const double coarse_dt,
   const hier::IntVector& ratio)
{
   NULL_USE(finer_level_number);

   TBOX_ASSERT(ratio.min() > 0);

   return coarse_dt / double(ratio.max());
}

void Problem::standardLevelSynchronization(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const int coarsest_level,
   const int finest_level,
   const double sync_time,
   const std::vector<double>& old_times)
{

}

void Problem::synchronizeNewLevels(
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const int coarsest_level,
   const int finest_level,
   const double sync_time,
   const bool initial_time)
{

    //Not needed, but not absolutely sure
    /*for (int fine_ln = finest_level; fine_ln > coarsest_level; --fine_ln) {
        const int coarse_ln = fine_ln - 1;
        std::shared_ptr<hier::PatchLevel> fine_level(hierarchy->getPatchLevel(fine_ln));
        d_bdry_fill_init->createSchedule(fine_level, coarse_ln, hierarchy, this)->fillData(sync_time, true);
    }*/
}

void Problem::resetTimeDependentData(
   const std::shared_ptr<hier::PatchLevel>& level,
   const double new_time,
   const bool can_be_refined)
{
   TBOX_ASSERT(level);
   level->setTime(new_time);
}

void Problem::resetDataToPreadvanceState(
   const std::shared_ptr<hier::PatchLevel>& level)
{
    //cout<<"resetDataToPreadvanceState"<<endl;
}

/*
 * Map data on a patch. This mapping is done only at the begining of the simulation.
 */
void Problem::mapDataOnPatch(const double time, const bool initial_time, const int ln, const std::shared_ptr< hier::PatchLevel >& level)
{
	(void) time;
	const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
   	if (initial_time || ln == 0) {

		// Mapping		
		int i, iterm, previousMapi, iWallAcc;
		bool interiorMapi;
		double iMapStart, iMapEnd;
		int j, jterm, previousMapj, jWallAcc;
		bool interiorMapj;
		double jMapStart, jMapEnd;
		int k, kterm, previousMapk, kWallAcc;
		bool interiorMapk;
		double kMapStart, kMapEnd;
		int minBlock[3], maxBlock[3], unionsI, facePointI, ie1, ie2, ie3, proc, pcounter, working, finished, pred;
		double maxDistance, e1, e2, e3;
		bool done, modif, workingArray[mpi.getSize()], finishedArray[mpi.getSize()], workingGlobal, finishedGlobal, workingPatchArray[level->getLocalNumberOfPatches()], finishedPatchArray[level->getLocalNumberOfPatches()], workingPatchGlobal, finishedPatchGlobal;
		int nodes = mpi.getSize();
		int patches = level->getLocalNumberOfPatches();

		double SQRT3INV = 1.0/sqrt(3.0);

		if (ln == 0) {
			//FOV initialization
			for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
				const std::shared_ptr< hier::Patch >& patch = *p_it;

				//Get the dimensions of the patch
				hier::Box pbox = patch->getBox();
				double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
				double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
				double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
				double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
				double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
				double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
				double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
				const hier::Index boxfirst = patch->getBox().lower();
				const hier::Index boxlast  = patch->getBox().upper();

				//Get delta spaces into an array. dx, dy, dz.
				const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
				const double* dx  = patch_geom->getDx();

				int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
				int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
				int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

				for (i = 0; i < ilast; i++) {
					for (j = 0; j < jlast; j++) {
						for (k = 0; k < klast; k++) {
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
			d_mapping_fill->createSchedule(level, level)->fillData(initial_time, true);

			//Region: mainI
			for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
				const std::shared_ptr< hier::Patch >& patch = *p_it;

				//Get the dimensions of the patch
				hier::Box pbox = patch->getBox();
				double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
				double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
				double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
				double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
				double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
				double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
				double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
				double* interior_i = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_i_id).get())->getPointer();
				double* interior_j = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_j_id).get())->getPointer();
				double* interior_k = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_k_id).get())->getPointer();
				double* interior = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
				int* nonSync = ((pdat::NodeData<int> *) patch->getPatchData(d_nonSync_regridding_tag_id).get())->getPointer();
				const hier::Index boxfirst = patch->getBox().lower();
				const hier::Index boxlast  = patch->getBox().upper();

				//Get delta spaces into an array. dx, dy, dz.
				const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
				const double* dx  = patch_geom->getDx();

				int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
				int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
				int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

				iMapStart = 0;
				iMapEnd = d_grid_geometry->getPhysicalDomain().front().numberCells()[0];
				jMapStart = 0;
				jMapEnd = d_grid_geometry->getPhysicalDomain().front().numberCells()[1];
				kMapStart = 0;
				kMapEnd = d_grid_geometry->getPhysicalDomain().front().numberCells()[2];
				for (i = iMapStart; i <= iMapEnd; i++) {
					for (j = jMapStart; j <= jMapEnd; j++) {
						for (k = kMapStart; k <= kMapEnd; k++) {
							if (i >= boxfirst(0) - d_ghost_width && i <= boxlast(0) + d_ghost_width && j >= boxfirst(1) - d_ghost_width && j <= boxlast(1) + d_ghost_width && k >= boxfirst(2) - d_ghost_width && k <= boxlast(2) + d_ghost_width) {
								vector(FOV_1, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 100;
								vector(FOV_xLower, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
								vector(FOV_xUpper, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
								vector(FOV_yLower, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
								vector(FOV_yUpper, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
								vector(FOV_zLower, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
								vector(FOV_zUpper, i - boxfirst(0) + d_ghost_width, j - boxfirst(1) + d_ghost_width, k - boxfirst(2) + d_ghost_width) = 0;
							}
						}
					}
				}
				//Check stencil
				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							vector(interior_i, i, j, k) = 0;
							vector(interior_j, i, j, k) = 0;
							vector(interior_k, i, j, k) = 0;
						}
					}
				}
				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							if (vector(FOV_1, i, j, k) > 0) {
								setStencilLimits(patch, i, j, k, d_FOV_1_id);
							}
						}
					}
				}
			}
			d_mapping_fill->createSchedule(level, level)->fillData(initial_time, true);
			for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
				const std::shared_ptr< hier::Patch >& patch = *p_it;

				//Get the dimensions of the patch
				hier::Box pbox = patch->getBox();
				double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
				double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
				double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
				double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
				double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
				double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
				double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
				double* interior_i = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_i_id).get())->getPointer();
				double* interior_j = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_j_id).get())->getPointer();
				double* interior_k = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_k_id).get())->getPointer();
				double* interior = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
				int* nonSync = ((pdat::NodeData<int> *) patch->getPatchData(d_nonSync_regridding_tag_id).get())->getPointer();
				const hier::Index boxfirst = patch->getBox().lower();
				const hier::Index boxlast  = patch->getBox().upper();

				//Get delta spaces into an array. dx, dy, dz.
				const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
				const double* dx  = patch_geom->getDx();

				int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
				int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
				int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							if ((abs(vector(interior_i, i, j, k)) > 1 || abs(vector(interior_j, i, j, k)) > 1 || abs(vector(interior_k, i, j, k)) > 1) && (i < d_ghost_width || i >= ilast - d_ghost_width || j < d_ghost_width || j >= jlast - d_ghost_width || k < d_ghost_width || k >= klast - d_ghost_width)) {
								checkStencil(patch, i, j, k, d_FOV_1_id);
							}
						}
					}
				}
				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							if (vector(interior_i, i, j, k) != 0 || vector(interior_j, i, j, k) != 0 || vector(interior_k, i, j, k) != 0) {
								vector(FOV_1, i, j, k) = 100;
								vector(FOV_xLower, i, j, k) = 0;
								vector(FOV_xUpper, i, j, k) = 0;
								vector(FOV_yLower, i, j, k) = 0;
								vector(FOV_yUpper, i, j, k) = 0;
								vector(FOV_zLower, i, j, k) = 0;
								vector(FOV_zUpper, i, j, k) = 0;
							}
						}
					}
				}
			}
			d_mapping_fill->createSchedule(level, level)->fillData(initial_time, true);

		}
		//Boundaries Mapping
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr< hier::Patch >& patch = *p_it;

			//Get the dimensions of the patch
			hier::Box pbox = patch->getBox();
			double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
			double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
			double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
			double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
			double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
			double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
			double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast  = patch->getBox().upper();

			//Get delta spaces into an array. dx, dy, dz.
			const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
			const double* dx  = patch_geom->getDx();

			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

			//z-Upper
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) {
				for (k = klast - d_ghost_width; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							vector(FOV_zUpper, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
						}
					}
				}
			}
			//z-Lower
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)) {
				for (k = 0; k < d_ghost_width; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							vector(FOV_zLower, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
			//y-Upper
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) {
				for (k = 0; k < klast; k++) {
					for (j = jlast - d_ghost_width; j < jlast; j++) {
						for (i = 0; i < ilast; i++) {
							vector(FOV_yUpper, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
			//y-Lower
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) {
				for (k = 0; k < klast; k++) {
					for (j = 0; j < d_ghost_width; j++) {
						for (i = 0; i < ilast; i++) {
							vector(FOV_yLower, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
			//x-Upper
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) {
				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = ilast - d_ghost_width; i < ilast; i++) {
							vector(FOV_xUpper, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xLower, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
			//x-Lower
			if (patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) {
				for (k = 0; k < klast; k++) {
					for (j = 0; j < jlast; j++) {
						for (i = 0; i < d_ghost_width; i++) {
							vector(FOV_xLower, i, j, k) = 100;
							vector(FOV_1, i, j, k) = 0;
							vector(FOV_xUpper, i, j, k) = 0;
							vector(FOV_yLower, i, j, k) = 0;
							vector(FOV_yUpper, i, j, k) = 0;
							vector(FOV_zLower, i, j, k) = 0;
							vector(FOV_zUpper, i, j, k) = 0;
						}
					}
				}
			}
		}
		d_mapping_fill->createSchedule(level, level)->fillData(initial_time, true);




   	}
}


/*
 * Sets the limit for the checkstencil routine
 */
void Problem::setStencilLimits(std::shared_ptr< hier::Patch > patch, int i, int j, int k, int v) const {
	double* FOV = ((pdat::NodeData<double> *) patch->getPatchData(v).get())->getPointer();
	double* interior_i = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_i_id).get())->getPointer();
	double* interior_j = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_j_id).get())->getPointer();
	double* interior_k = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_k_id).get())->getPointer();
	//Get the dimensions of the patch
	const hier::Index boxfirst = patch->getBox().lower();
	const hier::Index boxlast  = patch->getBox().upper();
	//Auxiliary definitions
	int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
	int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
	int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

	int iStart, iEnd, currentGhosti, otherSideShifti, jStart, jEnd, currentGhostj, otherSideShiftj, kStart, kEnd, currentGhostk, otherSideShiftk, shift;

	currentGhosti = d_ghost_width - 1;
	otherSideShifti = 0;
	currentGhostj = d_ghost_width - 1;
	otherSideShiftj = 0;
	currentGhostk = d_ghost_width - 1;
	otherSideShiftk = 0;
	//Checking width
	if ((i + 1 < ilast && vector(FOV, i + 1, j, k) == 0) ||  (i - 1 >= 0 && vector(FOV, i - 1, j, k) == 0)) {
		if (i + 1 < ilast && vector(FOV, i + 1, j, k) > 0) {
			bool stop_counting = false;
			for(int iti = i + 1; iti <= i + d_ghost_width - 1 && currentGhosti > 0; iti++) {
				if (iti < ilast  && vector(FOV, iti, j, k) > 0 && stop_counting == false) {
					currentGhosti--;
				} else {
					//First not interior point found
					if (iti < ilast  && vector(FOV, iti, j, k) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (iti >= ilast - 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (0, 1)) {
						stop_counting = true;
						//Calculate the number of cells the limit cannot grow
						if (iti + currentGhosti/2 >= ilast - 3) {
							otherSideShifti = (iti  + currentGhosti/2) - (ilast - 3);
						}
					}
				}
			}
		}
		if (i - 1 >= 0 && vector(FOV, i - 1, j, k) > 0) {
			bool stop_counting = false;
			for(int iti = i - 1; iti >= i - d_ghost_width + 1 && currentGhosti > 0; iti--) {
				if (iti >= 0  && vector(FOV, iti, j, k) > 0) {
					currentGhosti--;
				} else {
					//First not interior point found
					if (iti >= 0 && vector(FOV, iti, j, k) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (iti < 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (0, 0)) {
						stop_counting = true;
						//calculate the number of cells the limit cannot grow
						if (iti  -  ((currentGhosti) - currentGhosti/2) < 3) {
							otherSideShifti = 3 - (iti  - ((currentGhosti) - currentGhosti/2));
						}
					}
				}
			}
		}
		if (currentGhosti > 0) {
			if (i + 1 < ilast && vector(FOV, i + 1, j, k) > 0) {
				shift = 0;
				if(patch->getPatchGeometry()->getTouchesRegularBoundary (0, 0)) {
					while(i - ((currentGhosti) - currentGhosti/2) + shift < 3) {
						shift++;
					}
				}
				iStart = (currentGhosti - currentGhosti/2) - shift - otherSideShifti;
				iEnd = 0;
			} else {
				if (i - 1 >= 0 && vector(FOV, i - 1, j, k) > 0) {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (0, 1)) {
						while(i + currentGhosti/2 + shift >= ilast - 3) {
							shift--;
						}
					}
					iStart = 0;
					iEnd = currentGhosti/2 + shift + otherSideShifti;
				} else {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (0, 0)) {
						while(i - ((currentGhosti) - currentGhosti/2) + shift < 3) {
							shift++;
						}
					}
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (0, 1)) {
						while(i + currentGhosti/2 + shift >= ilast - 3) {
							shift--;
						}
					}
					iStart = (currentGhosti - currentGhosti/2) - shift;
					iEnd = currentGhosti/2 + shift;
				}
			}
		} else {
			iStart = 0;
			iEnd = 0;
		}
	} else {
		iStart = 0;
		iEnd = 0;
	}
	if ((j + 1 < jlast && vector(FOV, i, j + 1, k) == 0) ||  (j - 1 >= 0 && vector(FOV, i, j - 1, k) == 0)) {
		if (j + 1 < jlast && vector(FOV, i, j + 1, k) > 0) {
			bool stop_counting = false;
			for(int itj = j + 1; itj <= j + d_ghost_width - 1 && currentGhostj > 0; itj++) {
				if (itj < jlast  && vector(FOV, i, itj, k) > 0 && stop_counting == false) {
					currentGhostj--;
				} else {
					//First not interior point found
					if (itj < jlast  && vector(FOV, i, itj, k) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (itj >= jlast - 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (1, 1)) {
						stop_counting = true;
						//Calculate the number of cells the limit cannot grow
						if (itj + currentGhostj/2 >= jlast - 3) {
							otherSideShiftj = (itj  + currentGhostj/2) - (jlast - 3);
						}
					}
				}
			}
		}
		if (j - 1 >= 0 && vector(FOV, i, j - 1, k) > 0) {
			bool stop_counting = false;
			for(int itj = j - 1; itj >= j - d_ghost_width + 1 && currentGhostj > 0; itj--) {
				if (itj >= 0  && vector(FOV, i, itj, k) > 0) {
					currentGhostj--;
				} else {
					//First not interior point found
					if (itj >= 0 && vector(FOV, i, itj, k) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (itj < 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (1, 0)) {
						stop_counting = true;
						//calculate the number of cells the limit cannot grow
						if (itj  -  ((currentGhostj) - currentGhostj/2) < 3) {
							otherSideShiftj = 3 - (itj  - ((currentGhostj) - currentGhostj/2));
						}
					}
				}
			}
		}
		if (currentGhostj > 0) {
			if (j + 1 < jlast && vector(FOV, i, j + 1, k) > 0) {
				shift = 0;
				if(patch->getPatchGeometry()->getTouchesRegularBoundary (1, 0)) {
					while(j - ((currentGhostj) - currentGhostj/2) + shift < 3) {
						shift++;
					}
				}
				jStart = (currentGhostj - currentGhostj/2) - shift - otherSideShiftj;
				jEnd = 0;
			} else {
				if (j - 1 >= 0 && vector(FOV, i, j - 1, k) > 0) {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (1, 1)) {
						while(j + currentGhostj/2 + shift >= jlast - 3) {
							shift--;
						}
					}
					jStart = 0;
					jEnd = currentGhostj/2 + shift + otherSideShiftj;
				} else {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (1, 0)) {
						while(j - ((currentGhostj) - currentGhostj/2) + shift < 3) {
							shift++;
						}
					}
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (1, 1)) {
						while(j + currentGhostj/2 + shift >= jlast - 3) {
							shift--;
						}
					}
					jStart = (currentGhostj - currentGhostj/2) - shift;
					jEnd = currentGhostj/2 + shift;
				}
			}
		} else {
			jStart = 0;
			jEnd = 0;
		}
	} else {
		jStart = 0;
		jEnd = 0;
	}
	if ((k + 1 < klast && vector(FOV, i, j, k + 1) == 0) ||  (k - 1 >= 0 && vector(FOV, i, j, k - 1) == 0)) {
		if (k + 1 < klast && vector(FOV, i, j, k + 1) > 0) {
			bool stop_counting = false;
			for(int itk = k + 1; itk <= k + d_ghost_width - 1 && currentGhostk > 0; itk++) {
				if (itk < klast  && vector(FOV, i, j, itk) > 0 && stop_counting == false) {
					currentGhostk--;
				} else {
					//First not interior point found
					if (itk < klast  && vector(FOV, i, j, itk) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (itk >= klast - 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (2, 1)) {
						stop_counting = true;
						//Calculate the number of cells the limit cannot grow
						if (itk + currentGhostk/2 >= klast - 3) {
							otherSideShiftk = (itk  + currentGhostk/2) - (klast - 3);
						}
					}
				}
			}
		}
		if (k - 1 >= 0 && vector(FOV, i, j, k - 1) > 0) {
			bool stop_counting = false;
			for(int itk = k - 1; itk >= k - d_ghost_width + 1 && currentGhostk > 0; itk--) {
				if (itk >= 0  && vector(FOV, i, j, itk) > 0) {
					currentGhostk--;
				} else {
					//First not interior point found
					if (itk >= 0 && vector(FOV, i, j, itk) == 0) {
						stop_counting = true;
					}
					//Physical boundary reach
					if (itk < 3 && patch->getPatchGeometry()->getTouchesRegularBoundary (2, 0)) {
						stop_counting = true;
						//calculate the number of cells the limit cannot grow
						if (itk  -  ((currentGhostk) - currentGhostk/2) < 3) {
							otherSideShiftk = 3 - (itk  - ((currentGhostk) - currentGhostk/2));
						}
					}
				}
			}
		}
		if (currentGhostk > 0) {
			if (k + 1 < klast && vector(FOV, i, j, k + 1) > 0) {
				shift = 0;
				if(patch->getPatchGeometry()->getTouchesRegularBoundary (2, 0)) {
					while(k - ((currentGhostk) - currentGhostk/2) + shift < 3) {
						shift++;
					}
				}
				kStart = (currentGhostk - currentGhostk/2) - shift - otherSideShiftk;
				kEnd = 0;
			} else {
				if (k - 1 >= 0 && vector(FOV, i, j, k - 1) > 0) {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (2, 1)) {
						while(k + currentGhostk/2 + shift >= klast - 3) {
							shift--;
						}
					}
					kStart = 0;
					kEnd = currentGhostk/2 + shift + otherSideShiftk;
				} else {
					shift = 0;
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (2, 0)) {
						while(k - ((currentGhostk) - currentGhostk/2) + shift < 3) {
							shift++;
						}
					}
					if(patch->getPatchGeometry()->getTouchesRegularBoundary (2, 1)) {
						while(k + currentGhostk/2 + shift >= klast - 3) {
							shift--;
						}
					}
					kStart = (currentGhostk - currentGhostk/2) - shift;
					kEnd = currentGhostk/2 + shift;
				}
			}
		} else {
			kStart = 0;
			kEnd = 0;
		}
	} else {
		kStart = 0;
		kEnd = 0;
	}
	//Assigning stencil limits
	for(int iti = i - iStart; iti <= i + iEnd; iti++) {
		for(int itj = j - jStart; itj <= j + jEnd; itj++) {
			for(int itk = k - kStart; itk <= k + kEnd; itk++) {
				if(iti >= 0 && iti < ilast && itj >= 0 && itj < jlast && itk >= 0 && itk < klast && vector(FOV, iti, itj, itk) == 0) {
					if (i - iti < 0) {
						vector(interior_i, iti, itj, itk) = - (iStart + 1) - (i - iti);
					} else {
						vector(interior_i, iti, itj, itk) = (iStart + 1) - (i - iti);
					}
					if (j - itj < 0) {
						vector(interior_j, iti, itj, itk) = - (jStart + 1) - (j - itj);
					} else {
						vector(interior_j, iti, itj, itk) = (jStart + 1) - (j - itj);
					}
					if (k - itk < 0) {
						vector(interior_k, iti, itj, itk) = - (kStart + 1) - (k - itk);
					} else {
						vector(interior_k, iti, itj, itk) = (kStart + 1) - (k - itk);
					}
				}
			}
		}
	}
}

/*
 * Checks if the point has a stencil width
 */
void Problem::checkStencil(std::shared_ptr< hier::Patch > patch, int i, int j, int k, int v) const {
	double* FOV = ((pdat::NodeData<double> *) patch->getPatchData(v).get())->getPointer();
	double* interior_i = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_i_id).get())->getPointer();
	double* interior_j = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_j_id).get())->getPointer();
	double* interior_k = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_k_id).get())->getPointer();
	//Get the dimensions of the patch
	const hier::Index boxfirst = patch->getBox().lower();
	const hier::Index boxlast  = patch->getBox().upper();
	//Auxiliary definitions
	int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
	int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
	int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

	int i_i = vector(interior_i, i, j, k);
	int iStart = MAX(0, i_i) - 1;
	int iEnd = MAX(0, -i_i) - 1;
	int i_j = vector(interior_j, i, j, k);
	int jStart = MAX(0, i_j) - 1;
	int jEnd = MAX(0, -i_j) - 1;
	int i_k = vector(interior_k, i, j, k);
	int kStart = MAX(0, i_k) - 1;
	int kEnd = MAX(0, -i_k) - 1;

	for(int iti = i - iStart; iti <= i + iEnd; iti++) {
		for(int itj = j - jStart; itj <= j + jEnd; itj++) {
			for(int itk = k - kStart; itk <= k + kEnd; itk++) {
				if(iti >= 0 && iti < ilast && itj >= 0 && itj < jlast && itk >= 0 && itk < klast && vector(FOV, iti, itj, itk) == 0) {
					vector(interior_i, iti, itj, itk) = i_i  - (i - iti);
					vector(interior_j, iti, itj, itk) = i_j  - (j - itj);
					vector(interior_k, iti, itj, itk) = i_k  - (k - itk);
				}
			}
		}
	}
}


// Point class for the floodfill algorithm
class Point {
private:
public:
	int i, j, k;
};

void Problem::floodfill(std::shared_ptr< hier::Patch > patch, int i, int j, int k, int pred, int seg) const {

	double* FOV;
	switch(seg) {
		case 1:
			FOV = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		break;
	}
	int* nonSync = ((pdat::NodeData<int> *) patch->getPatchData(d_nonSync_regridding_tag_id).get())->getPointer();
	double* interior = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
	//Get the dimensions of the patch
	const hier::Index boxfirst = patch->getBox().lower();
	const hier::Index boxlast  = patch->getBox().upper();
	//Auxiliary definitions
	int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
	int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
	int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

	stack<Point> mystack;

	Point p;
	p.i = i;
	p.j = j;
	p.k = k;

	mystack.push(p);
	while(mystack.size() > 0) {
		p = mystack.top();
		mystack.pop();
		if (vector(nonSync, p.i, p.j, p.k) == 0) {
			vector(nonSync, p.i, p.j, p.k) = pred;
			vector(interior, p.i, p.j, p.k) = pred;
			if (pred == 2) {
				vector(FOV, p.i, p.j, p.k) = 100;
			}
			if (p.i - 1 >= 0 && vector(nonSync, p.i - 1, p.j, p.k) == 0) {
				Point np;
				np.i = p.i-1;
				np.j = p.j;
				np.k = p.k;
				mystack.push(np);
			}
			if (p.i + 1 < ilast && vector(nonSync, p.i + 1, p.j, p.k) == 0) {
				Point np;
				np.i = p.i+1;
				np.j = p.j;
				np.k = p.k;
				mystack.push(np);
			}
			if (p.j - 1 >= 0 && vector(nonSync, p.i, p.j - 1, p.k) == 0) {
				Point np;
				np.i = p.i;
				np.j = p.j-1;
				np.k = p.k;
				mystack.push(np);
			}
			if (p.j + 1 < jlast && vector(nonSync, p.i, p.j + 1, p.k) == 0) {
				Point np;
				np.i = p.i;
				np.j = p.j+1;
				np.k = p.k;
				mystack.push(np);
			}
			if (p.k - 1 >= 0 && vector(nonSync, p.i, p.j, p.k - 1) == 0) {
				Point np;
				np.i = p.i;
				np.j = p.j;
				np.k = p.k-1;
				mystack.push(np);
			}
			if (p.k + 1 < klast && vector(nonSync, p.i, p.j, p.k + 1) == 0) {
				Point np;
				np.i = p.i;
				np.j = p.j;
				np.k = p.k+1;
				mystack.push(np);
			}
		}
	}
}


/*
 * FOV correction for AMR
 */
void Problem::correctFOVS(const std::shared_ptr< hier::PatchLevel >& level) {
	int i, j, k;
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr< hier::Patch >& patch = *p_it;

		//Get the dimensions of the patch
		hier::Box pbox = patch->getBox();
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast  = patch->getBox().upper();

		//Get delta spaces into an array. dx, dy, dz.
		const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();

		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

		for (i = 0; i < ilast; i++) {
			for (j = 0; j < jlast; j++) {
				for (k = 0; k < klast; k++) {
					if (vector(FOV_xLower, i, j, k) > 0) {
						vector(FOV_xLower, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
					if (vector(FOV_xUpper, i, j, k) > 0) {
						vector(FOV_xUpper, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
					if (vector(FOV_yLower, i, j, k) > 0) {
						vector(FOV_yLower, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
					if (vector(FOV_yUpper, i, j, k) > 0) {
						vector(FOV_yUpper, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
					if (vector(FOV_zLower, i, j, k) > 0) {
						vector(FOV_zLower, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
					if (vector(FOV_zUpper, i, j, k) > 0) {
						vector(FOV_zUpper, i, j, k) = 100;
						vector(FOV_1, i, j, k) = 0;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
					}
					if (vector(FOV_1, i, j, k) > 0) {
						vector(FOV_1, i, j, k) = 100;
						vector(FOV_xLower, i, j, k) = 0;
						vector(FOV_xUpper, i, j, k) = 0;
						vector(FOV_yLower, i, j, k) = 0;
						vector(FOV_yUpper, i, j, k) = 0;
						vector(FOV_zLower, i, j, k) = 0;
						vector(FOV_zUpper, i, j, k) = 0;
					}
				}
			}
		}
	}
}




void Problem::interphaseMapping(const double time ,const bool initial_time,const int ln, const std::shared_ptr< hier::PatchLevel >& level, const int remesh) {
	const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
	if (remesh == 1) {
		//Update extrapolation variables
		//Calculation of hard region distance variables
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr< hier::Patch >& patch = *p_it;

			//Get the dimensions of the patch
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast  = patch->getBox().upper();
			const hier::IntVector ratio = level->getRatioToCoarserLevel();
			double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
			double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
			double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
			double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
			double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
			double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
			double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
			double* stalled_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_stalled_1_id).get())->getPointer();
			//Hard region field distance variables
			double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();

			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
			for (int i = 0; i < ilast; i++) {
				for (int j = 0; j < jlast; j++) {
					for (int k = 0; k < klast; k++) {
						vector(stalled_1, i, j, k) = checkStalled(patch, i, j, k, d_FOV_1_id);
						vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = 0;
						vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = 0;
						vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = 0;
					}
				}
			}
		}
		d_mapping_fill->createSchedule(level, level)->fillData(time, true);
		for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
			const std::shared_ptr< hier::Patch >& patch = *p_it;

			//Get the dimensions of the patch
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast  = patch->getBox().upper();
			const hier::IntVector ratio = level->getRatioToCoarserLevel();
			double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
			double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
			double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
			double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
			double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
			double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
			double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
			double* stalled_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_stalled_1_id).get())->getPointer();
			//Hard region field distance variables
			double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
			double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();

			int reGrid_i;
			if (ratio(0) == 0) reGrid_i = 1;
			else           reGrid_i = ratio(0);
			int reGrid_j;
			if (ratio(1) == 0) reGrid_j = 1;
			else           reGrid_j = ratio(1);
			int reGrid_k;
			if (ratio(2) == 0) reGrid_k = 1;
			else           reGrid_k = ratio(2);

			int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
			int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
			int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
			int dist_i_tmp, dist_i_tmp_p, dist_i_tmp_m, dist_i_tmp_r;
			int dist_j_tmp, dist_j_tmp_p, dist_j_tmp_m, dist_j_tmp_r;
			int dist_k_tmp, dist_k_tmp_p, dist_k_tmp_m, dist_k_tmp_r;
			int dist_r, dist_r_tmp;

			for (int i = 0; i < ilast; i++) {
				for (int j = 0; j < jlast; j++) {
					for (int k = 0; k < klast; k++) {
						if (vector(FOV_zUpper, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
						if (vector(FOV_zLower, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
						if (vector(FOV_yUpper, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
						if (vector(FOV_yLower, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
						if (vector(FOV_xUpper, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
						if (vector(FOV_xLower, i, j, k) > 0) {
							dist_i_tmp = 999;
							dist_i_tmp_p = 999;
							dist_i_tmp_m = 999;
							dist_i_tmp_r = 999;
							dist_j_tmp = 999;
							dist_j_tmp_p = 999;
							dist_j_tmp_m = 999;
							dist_j_tmp_r = 999;
							dist_k_tmp = 999;
							dist_k_tmp_p = 999;
							dist_k_tmp_m = 999;
							dist_k_tmp_r = 999;
							dist_r = 999;
							for (int Dist_i = -3 * reGrid_i; Dist_i <= 3 * reGrid_i; Dist_i++) {
								for (int Dist_j = -3 * reGrid_j; Dist_j <= 3 * reGrid_j; Dist_j++) {
									for (int Dist_k = -3 * reGrid_k; Dist_k <= 3 * reGrid_k; Dist_k++) {
										if (i + Dist_i >= 0 && i + Dist_i < ilast && j + Dist_j >= 0 && j + Dist_j < jlast && k + Dist_k >= 0 && k + Dist_k < klast && (!vector(stalled_1, i + Dist_i, j + Dist_j, k + Dist_k))) {
											if (Dist_i < 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_m)) {
												dist_i_tmp_m = -Dist_i;
											}
											if (Dist_i > 0 && Dist_j == 0 && Dist_k == 0 && fabs(Dist_i) < fabs(dist_i_tmp_p)) {
												dist_i_tmp_p = -Dist_i;
											}
											if (Dist_j < 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_m)) {
												dist_j_tmp_m = -Dist_j;
											}
											if (Dist_j > 0 && Dist_i == 0 && Dist_k == 0 && fabs(Dist_j) < fabs(dist_j_tmp_p)) {
												dist_j_tmp_p = -Dist_j;
											}
											if (Dist_k < 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_m)) {
												dist_k_tmp_m = -Dist_k;
											}
											if (Dist_k > 0 && Dist_i == 0 && Dist_j == 0 && fabs(Dist_k) < fabs(dist_k_tmp_p)) {
												dist_k_tmp_p = -Dist_k;
											}
											dist_r_tmp = Dist_i * Dist_i + Dist_j * Dist_j + Dist_k * Dist_k;
											if (dist_r_tmp < dist_r) {
												dist_r = dist_r_tmp;
												dist_i_tmp_r = -Dist_i;
												dist_j_tmp_r = -Dist_j;
												dist_k_tmp_r = -Dist_k;
											}
										}
									}
								}
							}
							if (fabs(dist_i_tmp_m) == fabs(dist_i_tmp_p) && dist_i_tmp_m < 999) {
								dist_i_tmp = 0;
								bool enough = true;
								for (int Dist_i = 0; Dist_i <= 2 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_p + Dist_i < ilast && (!vector(stalled_1, i - dist_i_tmp_p + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_p;
								}
								enough = true;
								for (int Dist_i = -2; Dist_i <= 0 && enough; Dist_i++) {
									if (!(i - dist_i_tmp_m + Dist_i >= 0 && (!vector(stalled_1, i - dist_i_tmp_m + Dist_i, j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_i_tmp = dist_i_tmp_m;
								}
							} else if (fabs(dist_i_tmp_m) < fabs(dist_i_tmp_p)) {
								dist_i_tmp = dist_i_tmp_m;
							} else {
								dist_i_tmp = dist_i_tmp_p;
							}
							if (fabs(dist_j_tmp_m) == fabs(dist_j_tmp_p) && dist_j_tmp_m < 999) {
								dist_j_tmp = 0;
								bool enough = true;
								for (int Dist_j = 0; Dist_j <= 2 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_p + Dist_j < jlast && (!vector(stalled_1, i, j - dist_j_tmp_p + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_p;
								}
								enough = true;
								for (int Dist_j = -2; Dist_j <= 0 && enough; Dist_j++) {
									if (!(j - dist_j_tmp_m + Dist_j >= 0 && (!vector(stalled_1, i, j - dist_j_tmp_m + Dist_j, k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_j_tmp = dist_j_tmp_m;
								}
							} else if (fabs(dist_j_tmp_m) < fabs(dist_j_tmp_p)) {
								dist_j_tmp = dist_j_tmp_m;
							} else {
								dist_j_tmp = dist_j_tmp_p;
							}
							if (fabs(dist_k_tmp_m) == fabs(dist_k_tmp_p) && dist_k_tmp_m < 999) {
								dist_k_tmp = 0;
								bool enough = true;
								for (int Dist_k = 0; Dist_k <= 2 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_p + Dist_k < klast && (!vector(stalled_1, i, j, k - dist_k_tmp_p + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_p;
								}
								enough = true;
								for (int Dist_k = -2; Dist_k <= 0 && enough; Dist_k++) {
									if (!(k - dist_k_tmp_m + Dist_k >= 0 && (!vector(stalled_1, i, j, k - dist_k_tmp_m + Dist_k)))) {
										enough = false;
									}
								}
								if (enough) {
									dist_k_tmp = dist_k_tmp_m;
								}
							} else if (fabs(dist_k_tmp_m) < fabs(dist_k_tmp_p)) {
								dist_k_tmp = dist_k_tmp_m;
							} else {
								dist_k_tmp = dist_k_tmp_p;
							}
							if (dist_i_tmp == 999 && dist_j_tmp == 999 && dist_k_tmp == 999) {
								dist_i_tmp = dist_i_tmp_r;
								dist_j_tmp = dist_j_tmp_r;
								dist_k_tmp = dist_k_tmp_r;
							}
							if (dist_i_tmp != 999 && ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_i_tmp) < fabs(vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_i_tmp;
							}
							if (dist_j_tmp != 999 && ((vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_j_tmp) < fabs(vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_j_tmp;
							}
							if (dist_k_tmp != 999 && ((vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) == 0) || (fabs(dist_k_tmp) < fabs(vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k))))) {
								vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) = dist_k_tmp;
							}
						}
					}
				}
			}
		}
		d_mapping_fill->createSchedule(level, level)->fillData(initial_time, true);
	}
}


/*
 * Checks if the point has to be stalled
 */
bool Problem::checkStalled(std::shared_ptr< hier::Patch > patch, int i, int j, int k, int v) const {
	double* FOV = ((pdat::NodeData<double> *) patch->getPatchData(v).get())->getPointer();
	double* interior = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
	//Get the dimensions of the patch
	const hier::Index boxfirst = patch->getBox().lower();
	const hier::Index boxlast  = patch->getBox().upper();
	//Auxiliary definitions
	int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
	int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
	int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;

	int stencilAcc, stencilAccMax_i, stencilAccMax_j, stencilAccMax_k;
	bool notEnoughStencil = false;
	int FOV_threshold = 0;
	if (vector(FOV, i, j, k) <= FOV_threshold) {
		notEnoughStencil = true;
	} else {
		stencilAcc = 0;
		stencilAccMax_i = 0;
		for (int it1 = MAX(i-d_regionMinThickness, 0); it1 <= MIN(i+d_regionMinThickness, ilast - 1); it1++) {
			if (vector(FOV, it1, j, k) > FOV_threshold) {
				stencilAcc++;
			} else {
				stencilAccMax_i = MAX(stencilAccMax_i, stencilAcc);
				stencilAcc = 0;
			}
		}
		stencilAccMax_i = MAX(stencilAccMax_i, stencilAcc);
		stencilAcc = 0;
		stencilAccMax_j = 0;
		for (int jt1 = MAX(j-d_regionMinThickness, 0); jt1 <= MIN(j+d_regionMinThickness, jlast - 1); jt1++) {
			if (vector(FOV, i, jt1, k) > FOV_threshold) {
				stencilAcc++;
			} else {
				stencilAccMax_j = MAX(stencilAccMax_j, stencilAcc);
				stencilAcc = 0;
			}
		}
		stencilAccMax_j = MAX(stencilAccMax_j, stencilAcc);
		stencilAcc = 0;
		stencilAccMax_k = 0;
		for (int kt1 = MAX(k-d_regionMinThickness, 0); kt1 <= MIN(k+d_regionMinThickness, klast - 1); kt1++) {
			if (vector(FOV, i, j, kt1) > FOV_threshold) {
				stencilAcc++;
			} else {
				stencilAccMax_k = MAX(stencilAccMax_k, stencilAcc);
				stencilAcc = 0;
			}
		}
		stencilAccMax_k = MAX(stencilAccMax_k, stencilAcc);
		if ((stencilAccMax_i < d_regionMinThickness) || (stencilAccMax_j < d_regionMinThickness) || (stencilAccMax_k < d_regionMinThickness)) {
			notEnoughStencil = true;
		}
	}
	return notEnoughStencil;
}






/*
 * Initialize data on a patch. This initialization is done only at the begining of the simulation.
 */
void Problem::initializeDataOnPatch(hier::Patch& patch, 
                                    const double time,
                                    const bool initial_time)
{
	(void) time;
   	if (initial_time) {
		// Initial conditions		
		//Get fields, auxiliary fields and local variables that are going to be used.
		double* Psisf = ((pdat::NodeData<double> *) patch.getPatchData(d_Psisf_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_Psisf_id).get())->fillAll(0);
		double* phi = ((pdat::NodeData<double> *) patch.getPatchData(d_phi_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_phi_id).get())->fillAll(0);
		double* Phid_x = ((pdat::NodeData<double> *) patch.getPatchData(d_Phid_x_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_Phid_x_id).get())->fillAll(0);
		double* Phid_y = ((pdat::NodeData<double> *) patch.getPatchData(d_Phid_y_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_Phid_y_id).get())->fillAll(0);
		double* Phid_z = ((pdat::NodeData<double> *) patch.getPatchData(d_Phid_z_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_Phid_z_id).get())->fillAll(0);
		double* Psi = ((pdat::NodeData<double> *) patch.getPatchData(d_Psi_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_Psi_id).get())->fillAll(0);
		double* phi2 = ((pdat::NodeData<double> *) patch.getPatchData(d_phi2_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_phi2_id).get())->fillAll(0);
		double* dphi2 = ((pdat::NodeData<double> *) patch.getPatchData(d_dphi2_id).get())->getPointer();
		((pdat::NodeData<double> *) patch.getPatchData(d_dphi2_id).get())->fillAll(0);
		double pos_x, pos_y, pos_z, r, drphi;
		
		//Get the dimensions of the patch
		hier::Box pbox = patch.getBox();
		double* FOV_1 = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch.getPatchData(d_FOV_zUpper_id).get())->getPointer();
		const hier::Index boxfirst = patch.getBox().lower();
		const hier::Index boxlast  = patch.getBox().upper();
		
		//Get delta spaces into an array. dx, dy, dz.
		const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch.getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
		
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if (vector(FOV_1, i, j, k) > 0) {
						if (equalsEq(Nstar, 1.0)) {
							pos_x = xcoord(i) - xcenter1;
							pos_y = ycoord(j) - ycenter1;
							pos_z = zcoord(k) - zcenter1;
							r = MAX(sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z), 0.0001);
							vector(phi, i, j, k) = 0.0;
							readFromFileQuintic(1.0, 1.0, r, vector(phi, i, j, k), 0.0, coord0_data_phir1d, coord0_dx_phir1d, coord0_size_phir1d, vars_data_phir1d, vars_size_phir1d);
							drphi = 0.0;
							readFromFileQuintic(1.0, 1.0, r, drphi, 0.0, coord0_data_drphir1d, coord0_dx_drphir1d, coord0_size_drphir1d, vars_data_drphir1d, vars_size_drphir1d);
							vector(Phid_z, i, j, k) = pos_z / r * drphi;
							vector(Phid_x, i, j, k) = pos_x / r * drphi;
							vector(Phid_y, i, j, k) = pos_y / r * drphi;
							vector(Psi, i, j, k) = 0.0;
							vector(Psisf, i, j, k) = nu * vector(phi, i, j, k);
							vector(phi2, i, j, k) = vector(phi, i, j, k) * vector(phi, i, j, k);
							vector(dphi2, i, j, k) = vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k) + vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k) + vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k);
						}
						if (equalsEq(Nstar, 2.0)) {
							pos_x = xcoord(i);
							pos_y = ycoord(j);
							pos_z = zcoord(k);
							r = MAX(sqrt(pos_x * pos_x + pos_y * pos_y + pos_z * pos_z), 0.0001);
							vector(phi, i, j, k) = 0.0;
							readFromFileQuintic(1.0, 1.0, r, vector(phi, i, j, k), 0.0, coord0_data_phir1d, coord0_dx_phir1d, coord0_size_phir1d, vars_data_phir1d, vars_size_phir1d);
							drphi = 0.0;
							readFromFileQuintic(1.0, 1.0, r, drphi, 0.0, coord0_data_drphir1d, coord0_dx_drphir1d, coord0_size_drphir1d, vars_data_drphir1d, vars_size_drphir1d);
							vector(Phid_z, i, j, k) = pos_z / r * drphi;
							vector(Phid_x, i, j, k) = pos_x / r * drphi;
							vector(Phid_y, i, j, k) = pos_y / r * drphi;
							vector(Psisf, i, j, k) = nu * vector(phi, i, j, k);
							vector(Psi, i, j, k) = 0.0;
							vector(phi2, i, j, k) = vector(phi, i, j, k) * vector(phi, i, j, k);
							vector(dphi2, i, j, k) = vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k) + vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k) + vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k);
						}
					}
		
				}
			}
		}
		

   	}
}



/*
 * Gets the coarser patch that contains the box
 */
const std::shared_ptr<hier::Patch >& Problem::getCoarserPatch(
	const std::shared_ptr< hier::PatchLevel >& level,
	const hier::Box interior, 
	const hier::IntVector ratio)
{
	const hier::Box& coarsenBox = hier::Box::coarsen(interior, ratio);
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr< hier::Patch >& patch = *p_it;
		const hier::Box& interior_C = patch->getBox();
		if (interior_C.intersects(coarsenBox)) {
			return patch;		
		}
	}
    return NULL;
}

/*
 * Reset the hierarchy-dependent internal information.
 */
void Problem::resetHierarchyConfiguration (
   const std::shared_ptr<hier::PatchHierarchy >& new_hierarchy ,
   int coarsest_level ,
   int finest_level )
{
	int finest_hiera_level = new_hierarchy->getFinestLevelNumber();

   	//  If we have added or removed a level, resize the schedule arrays

	d_bdry_sched_advance1.resize(finest_hiera_level+1);
	d_bdry_sched_advance7.resize(finest_hiera_level+1);
	d_bdry_sched_advance13.resize(finest_hiera_level+1);
	d_bdry_sched_advance19.resize(finest_hiera_level+1);
	d_bdry_sched_analysis1.resize(finest_hiera_level+1);
	d_coarsen_schedule.resize(finest_hiera_level+1);
	d_bdry_sched_postCoarsen.resize(finest_hiera_level+1);
	//  Build coarsen and refine communication schedules.
	for (int ln = coarsest_level; ln <= finest_hiera_level; ln++) {
		std::shared_ptr< hier::PatchLevel > level(new_hierarchy->getPatchLevel(ln));
		d_bdry_sched_advance1[ln] = d_bdry_fill_advance1->createSchedule(level,ln-1,new_hierarchy,this);
		d_bdry_sched_advance7[ln] = d_bdry_fill_advance7->createSchedule(level,ln-1,new_hierarchy,this);
		d_bdry_sched_advance13[ln] = d_bdry_fill_advance13->createSchedule(level,ln-1,new_hierarchy,this);
		d_bdry_sched_advance19[ln] = d_bdry_fill_advance19->createSchedule(level,ln-1,new_hierarchy,this);
		d_bdry_sched_analysis1[ln] = d_bdry_fill_analysis1->createSchedule(level,this);
		d_bdry_sched_postCoarsen[ln] = d_bdry_post_coarsen->createSchedule(level);
		// coarsen schedule only for levels > 0
		if (ln > 0) {
			std::shared_ptr< hier::PatchLevel > coarser_level(new_hierarchy->getPatchLevel(ln-1));
			d_coarsen_schedule[ln] = d_coarsen_algorithm->createSchedule(coarser_level, level, NULL);
		}
	}

}

/* 
 * Calculation of auxiliary terms after a new level is regridded.
 */
void Problem::postNewLevel(const std::shared_ptr< hier::PatchLevel >& level) {
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr< hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
		double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
		double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
		double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();

		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
		const double simPlat_dt = patch->getPatchData(d_FOV_1_id)->getTime();

		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((vector(FOV_1, i, j, k) > 0)) {
						vector(phi2, i, j, k) = vector(phi, i, j, k) * vector(phi, i, j, k);
						vector(dphi2, i, j, k) = vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k) + vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k) + vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k);
					}
				}
			}
		}
	}
}

/*
 * This method sets the physical boundary conditions.
 */
void Problem::setPhysicalBoundaryConditions(
   hier::Patch& patch,
   const double fill_time,
   const hier::IntVector& ghost_width_to_fill)
{
	//Boundary must not be implemented in this method
}

/*
 * Set up external plotter to plot internal data from this class.
 * Register variables appropriate for plotting.
 */
int Problem::setupPlotterMesh(appu::VisItDataWriter &plotter) const {
	if (!d_patch_hierarchy) {
		TBOX_ERROR(d_object_name << ": No hierarchy inn"
			<< " Problem::setupPlottern"
			<< "The hierarchy must be set before callingn"
			<< "this function.n");
	}
	plotter.registerPlotQuantity("FOV_1","SCALAR",d_FOV_1_id);
	hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
	for (set<string>::const_iterator it = d_full_mesh_writer_variables.begin() ; it != d_full_mesh_writer_variables.end(); ++it) {
		string var_to_register = *it;
		if (!(vdb->checkVariableExists(var_to_register))) {
			TBOX_ERROR(d_object_name << ": Variable selected for 3D write not found:" <<  var_to_register);
		}
		int var_id = vdb->getVariable(var_to_register)->getInstanceIdentifier();
		plotter.registerPlotQuantity(var_to_register,"SCALAR",var_id);
	}
	return 0;
}
/*
 * Set up external plotter to plot sliced data from this class.
 * Register variables appropriate for plotting.
 */
int Problem::setupSlicePlotter(vector<std::shared_ptr<SlicerDataWriter> > &plotters) const {
	if (!d_patch_hierarchy) {
	TBOX_ERROR(d_object_name << ": No hierarchy in\n"
		<< " Problem::setupSlicePlotter\n"
		<< "The hierarchy must be set before calling\n"
		<< "this function.\n");
	}
	int i = 0;
	for (vector<std::shared_ptr<SlicerDataWriter> >::const_iterator it = plotters.begin(); it != plotters.end(); ++it) {
		std::shared_ptr<SlicerDataWriter> plotter = *it;
		plotter->registerPlotQuantity("FOV_1","SCALAR",d_FOV_1_id);
		hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
		set<string> variables = d_sliceVariables[i];
		for (set<string>::const_iterator it2 = variables.begin() ; it2 != variables.end(); ++it2) {
			string var_to_register = *it2;
			if (!(vdb->checkVariableExists(var_to_register))) {
				TBOX_ERROR(d_object_name << ": Variable selected for Slice not found:" <<  var_to_register);
			}
			int var_id = vdb->getVariable(var_to_register)->getInstanceIdentifier();
			plotter->registerPlotQuantity(var_to_register,"SCALAR",var_id);
		}
		i++;
	}

	return 0;
}

/*
 * Set up external plotter to plot spherical data from this class.
 * Register variables appropriate for plotting.
 */
int Problem::setupSpherePlotter(vector<std::shared_ptr<SphereDataWriter> > &plotters) const {
	if (!d_patch_hierarchy) {
	TBOX_ERROR(d_object_name << ": No hierarchy in\n"
		<< " Problem::setupSpherePlotter\n"
		<< "The hierarchy must be set before calling\n"
		<< "this function.\n");
	}
	int i = 0;
	for (vector<std::shared_ptr<SphereDataWriter> >::const_iterator it = plotters.begin(); it != plotters.end(); ++it) {
		std::shared_ptr<SphereDataWriter> plotter = *it;
		plotter->registerPlotQuantity("FOV_1","SCALAR",d_FOV_1_id);
		hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
		set<string> variables = d_sphereVariables[i];
		for (set<string>::const_iterator it2 = variables.begin() ; it2 != variables.end(); ++it2) {
			string var_to_register = *it2;
			if (!(vdb->checkVariableExists(var_to_register))) {
				TBOX_ERROR(d_object_name << ": Variable selected for Sphere not found:" <<  var_to_register);
			}
			int var_id = vdb->getVariable(var_to_register)->getInstanceIdentifier();
			plotter->registerPlotQuantity(var_to_register,"SCALAR",var_id);
		}
		i++;
	}

	return 0;
}
/*
 * Set up external plotter to plot integration data from this class.
 * Register variables appropriate for plotting.
 */
int Problem::setupIntegralPlotter(vector<std::shared_ptr<IntegrateDataWriter> > &plotters) const {
	if (!d_patch_hierarchy) {
		TBOX_ERROR(d_object_name << ": No hierarchy inn"
		<< " Problem::setupIntegralPlottern"
		<< "The hierarchy must be set before callingn"
		<< "this function.n");
	}
	int i = 0;
	for (vector<std::shared_ptr<IntegrateDataWriter> >::const_iterator it = plotters.begin(); it != plotters.end(); ++it) {
		std::shared_ptr<IntegrateDataWriter> plotter = *it;
		hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
		set<string> variables = d_integralVariables[i];
		for (set<string>::const_iterator it2 = variables.begin() ; it2 != variables.end(); ++it2) {
			string var_to_register = *it2;
			if (!(vdb->checkVariableExists(var_to_register))) {
				TBOX_ERROR(d_object_name << ": Variable selected for Integration not found:" <<  var_to_register);
			}
			int var_id = vdb->getVariable(var_to_register)->getInstanceIdentifier();
			plotter->registerPlotQuantity(var_to_register,"SCALAR",var_id);
		}
		i++;
	}
	return 0;
}
/*
 * Set up external plotter to plot point data from this class.
 * Register variables appropriate for plotting.
 */
int Problem::setupPointPlotter(vector<std::shared_ptr<PointDataWriter> > &plotters) const {
	int i = 0;
	for (vector<std::shared_ptr<PointDataWriter> >::const_iterator it = plotters.begin(); it != plotters.end(); ++it) {
		std::shared_ptr<PointDataWriter> plotter = *it;
		hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
		set<string> variables = d_pointVariables[i];
		for (set<string>::const_iterator it2 = variables.begin() ; it2 != variables.end(); ++it2) {
			string var_to_register = *it2;
			if (!(vdb->checkVariableExists(var_to_register))) {
				TBOX_ERROR(d_object_name << ": Variable selected for Point not found:" <<  var_to_register);
			}
			int var_id = vdb->getVariable(var_to_register)->getInstanceIdentifier();
			plotter->registerPlotQuantity(var_to_register,"SCALAR",var_id);
		}
		i++;
	}
	return 0;
}


/*
 * Perform a single step from the discretization schema algorithm   
 */
double Problem::advanceLevel(
   const std::shared_ptr<hier::PatchLevel>& level,
   const std::shared_ptr<hier::PatchHierarchy>& hierarchy,
   const double current_time,
   const double new_time,
   const bool first_step,
   const bool last_step,
   const bool regrid_advance)
{
	const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

	const int ln = level->getLevelNumber();
	const double simPlat_dt = new_time - current_time;
	const double level_ratio = level->getRatioToCoarserLevel().max();
	if (first_step) {
		bo_substep_iteration[ln] = 0;
	}
	else {
		bo_substep_iteration[ln] = bo_substep_iteration[ln] + 1;
	}
	time_interpolate_operator_mesh1->setRatio(level_ratio);
	time_interpolate_operator_mesh1->setStep(0);
	time_interpolate_operator_mesh1->setTimeSubstepNumber(bo_substep_iteration[ln]);

	if (d_refinedTimeStepping && first_step && ln > 0) {
		current_iteration[ln] = (current_iteration[ln - 1] - 1) * hierarchy->getRatioToCoarserLevel(ln).max() + 1;
	} else {
		current_iteration[ln] = current_iteration[ln] + 1;
	}
	int previous_iteration = current_iteration[ln] - 1;
	int outputCycle = current_iteration[ln];
	int maxLevels = hierarchy->getMaxNumberOfLevels();
	if (maxLevels > ln + 1) {
		int currentLevelNumber = ln;
		while (currentLevelNumber < maxLevels - 1) {
			int ratio = hierarchy->getRatioToCoarserLevel(currentLevelNumber + 1).max();
			outputCycle = outputCycle * ratio;
			previous_iteration = previous_iteration * ratio;
			currentLevelNumber++;
		}
	}

	t_step->start();
  	// Shifting time
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch > patch = *p_it;
		std::shared_ptr< pdat::NodeData<double> > phi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_phi_id)));
		std::shared_ptr< pdat::NodeData<double> > phi_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_phi_p_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_z(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_z_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_z_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_z_p_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_x(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_x_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_x_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_x_p_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_y(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_y_id)));
		std::shared_ptr< pdat::NodeData<double> > Phid_y_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Phid_y_p_id)));
		std::shared_ptr< pdat::NodeData<double> > Psisf(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Psisf_id)));
		std::shared_ptr< pdat::NodeData<double> > Psisf_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Psisf_p_id)));
		std::shared_ptr< pdat::NodeData<double> > Psi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Psi_id)));
		std::shared_ptr< pdat::NodeData<double> > Psi_p(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_Psi_p_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1phi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1phi_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1Phid_z(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1Phid_z_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1Phid_x(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1Phid_x_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1Phid_y(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1Phid_y_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1Psisf(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1Psisf_id)));
		std::shared_ptr< pdat::NodeData<double> > rk1Psi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk1Psi_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2phi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2phi_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2Phid_z(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2Phid_z_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2Phid_x(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2Phid_x_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2Phid_y(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2Phid_y_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2Psisf(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2Psisf_id)));
		std::shared_ptr< pdat::NodeData<double> > rk2Psi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk2Psi_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3phi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3phi_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3Phid_z(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3Phid_z_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3Phid_x(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3Phid_x_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3Phid_y(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3Phid_y_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3Psisf(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3Psisf_id)));
		std::shared_ptr< pdat::NodeData<double> > rk3Psi(SAMRAI_SHARED_PTR_CAST<pdat::NodeData<double>, hier::PatchData>(patch->getPatchData(d_rk3Psi_id)));
		Psi_p->copy(*Psi);
		Psisf_p->copy(*Psisf);
		Phid_y_p->copy(*Phid_y);
		Phid_x_p->copy(*Phid_x);
		Phid_z_p->copy(*Phid_z);
		phi_p->copy(*phi);
		phi_p->setTime(current_time);
		Phid_z_p->setTime(current_time);
		Phid_x_p->setTime(current_time);
		Phid_y_p->setTime(current_time);
		Psisf_p->setTime(current_time);
		Psi_p->setTime(current_time);
		rk1phi->setTime(current_time + simPlat_dt * 0.5);
		rk1Phid_z->setTime(current_time + simPlat_dt * 0.5);
		rk1Phid_x->setTime(current_time + simPlat_dt * 0.5);
		rk1Phid_y->setTime(current_time + simPlat_dt * 0.5);
		rk1Psisf->setTime(current_time + simPlat_dt * 0.5);
		rk1Psi->setTime(current_time + simPlat_dt * 0.5);
		rk2phi->setTime(current_time + simPlat_dt * 0.5);
		rk2Phid_z->setTime(current_time + simPlat_dt * 0.5);
		rk2Phid_x->setTime(current_time + simPlat_dt * 0.5);
		rk2Phid_y->setTime(current_time + simPlat_dt * 0.5);
		rk2Psisf->setTime(current_time + simPlat_dt * 0.5);
		rk2Psi->setTime(current_time + simPlat_dt * 0.5);
		rk3phi->setTime(current_time + simPlat_dt);
		rk3Phid_z->setTime(current_time + simPlat_dt);
		rk3Phid_x->setTime(current_time + simPlat_dt);
		rk3Phid_y->setTime(current_time + simPlat_dt);
		rk3Psisf->setTime(current_time + simPlat_dt);
		rk3Psi->setTime(current_time + simPlat_dt);
		phi->setTime(current_time + simPlat_dt);
		Phid_z->setTime(current_time + simPlat_dt);
		Phid_x->setTime(current_time + simPlat_dt);
		Phid_y->setTime(current_time + simPlat_dt);
		Psisf->setTime(current_time + simPlat_dt);
		Psi->setTime(current_time + simPlat_dt);
	}
  	// Evolution
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* Psi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_p_id).get())->getPointer();
		double* Phid_x_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_p_id).get())->getPointer();
		double* Phid_y_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_p_id).get())->getPointer();
		double* Phid_z_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_p_id).get())->getPointer();
		double* phi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_p_id).get())->getPointer();
		double* Psisf_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_p_id).get())->getPointer();
		double* rk1Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psisf_id).get())->getPointer();
		double* rk1phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1phi_id).get())->getPointer();
		double* rk1Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_x_id).get())->getPointer();
		double* rk1Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_y_id).get())->getPointer();
		double* rk1Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_z_id).get())->getPointer();
		double* rk1Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psi_id).get())->getPointer();
		double d_Phid_x_o0_t0_m0_l0, d_Phid_y_o0_t0_m0_l0, d_Phid_z_o0_t0_m0_l0, d_Psi_o0_t0_m0_l0, d_Psi_o0_t1_m0_l0, d_Psi_o0_t2_m0_l0, d_Psi_o0_t4_m0_l0, d_Psi_o0_t5_m0_l0, d_Psi_o0_t7_m0_l0, Phiu_x_p, Phiu_z_p, Phiu_y_p, Xphi_p, r1_z_p, r2_z_p, rorbit_dynamic_p, omegaorbit_p, r2_y_p, r2_x_p, sq_r2_p, r1_y_p, r1_x_p, sq_r1_p, Tbar_p, gammax_off, KXXf_p, KXf_p, d_Psisf_o0_t0_m0_l0, d_phi_o0_t0_m0_l0, d_Psi_o0_t12_m0_l0, m_Psi_o0_t0_l0, m_Psi_o0_t1_l0, m_Psi_o0_t2_l0, m_Psi_o0_t3_l0, m_Psi_o0_t4_l0, m_Psi_o0_t5_l0, m_Psi_o0_t6_l0, m_Psi_o0_t7_l0, m_Psi_o0_t8_l0, m_Psi_o0_t9_l0, m_Psi_o0_t10_l0, m_Psi_o0_t11_l0, RHS_Psisf, RHS_phi, RHS_Phid_x, RHS_Phid_y, RHS_Phid_z, RHS_Psi, n_x, n_y, n_z, interaction_index, mod_normal, i_d_Phid_y_o0_t0_m0_l0, i_d_Phid_z_o0_t0_m0_l0, i_d_Phid_z_o0_t1_m0_l0, i_d_phi_o0_t0_m0_l0, i_d_phi_o0_t1_m0_l0, i_d_phi_o0_t2_m0_l0, i_d_Psisf_o0_t0_m0_l0, i_d_Psisf_o0_t1_m0_l0, i_d_Psisf_o0_t2_m0_l0, i_d_phi_o0_t3_m0_l0, i_d_Phid_x_o0_t3_m0_l0, i_d_Phid_y_o0_t3_m0_l0, i_d_Phid_z_o0_t3_m0_l0, i_d_Psisf_o0_t3_m0_l0, i_d_Psi_o0_t3_m0_l0, i_m_phi_o0_t0_l0, i_m_phi_o0_t1_l0, i_m_phi_o0_t2_l0, i_m_Phid_x_o0_t0_l0, i_m_Phid_x_o0_t1_l0, i_m_Phid_x_o0_t2_l0, i_m_Phid_y_o0_t0_l0, i_m_Phid_y_o0_t1_l0, i_m_Phid_y_o0_t2_l0, i_m_Phid_z_o0_t0_l0, i_m_Phid_z_o0_t1_l0, i_m_Phid_z_o0_t2_l0, i_m_Psisf_o0_t0_l0, i_m_Psisf_o0_t1_l0, i_m_Psisf_o0_t2_l0, i_m_Psi_o0_t0_l0, i_m_Psi_o0_t1_l0, i_m_Psi_o0_t2_l0;
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((i + 3 < ilast && i - 3 >= 0 && j + 3 < jlast && j - 3 >= 0 && k + 3 < klast && k - 3 >= 0)) {
						d_Phid_x_o0_t0_m0_l0 = D1CDO4_i(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_y_o0_t0_m0_l0 = D1CDO4_j(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_z_o0_t0_m0_l0 = D1CDO4_k(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t0_m0_l0 = D1CDO4_i(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t1_m0_l0 = D1CDO4_j(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t2_m0_l0 = D1CDO4_k(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t4_m0_l0 = D1CDO4_j(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t5_m0_l0 = D1CDO4_k(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t7_m0_l0 = D1CDO4_k(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast);
						Phiu_x_p = vector(Phid_x_p, i, j, k);
						Phiu_z_p = vector(Phid_z_p, i, j, k);
						Phiu_y_p = vector(Phid_y_p, i, j, k);
						Xphi_p = (-vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) + Phiu_x_p * vector(Phid_x_p, i, j, k) + Phiu_y_p * vector(Phid_y_p, i, j, k) + Phiu_z_p * vector(Phid_z_p, i, j, k);
						r1_z_p = zcoord(k);
						r2_z_p = zcoord(k);
						if (lessEq(current_time, torbit)) {
							rorbit_dynamic_p = rstart;
						}
						if ((current_time > torbit) && (current_time < (torbit + tslow))) {
							rorbit_dynamic_p = rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0);
						}
						if (greaterEq(current_time, (torbit + tslow))) {
							rorbit_dynamic_p = rorbit;
						}
						if (lessEq(current_time, tini)) {
							omegaorbit_p = 0.0;
						}
						if ((current_time > tini) && (current_time < (torbit + tini))) {
							omegaorbit_p = p_Omega * sqrt(2.0 * mass / ((2.0 * rorbit_dynamic_p) * (2.0 * rorbit_dynamic_p) * (2.0 * rorbit_dynamic_p))) * 4.0 * (2.0 * torbit - (current_time - tini)) * (current_time - tini) / ((2.0 * torbit) * (2.0 * torbit));
						}
						if (greaterEq(current_time, (tini + torbit))) {
							omegaorbit_p = p_Omega * sqrt(2.0 * mass / ((2.0 * rorbit_dynamic_p) * (2.0 * rorbit_dynamic_p) * (2.0 * rorbit_dynamic_p)));
						}
						r2_y_p = ycoord(j) - 2.0 / (1.0 + mu) * rorbit_dynamic_p * sin(omegaorbit_p * (current_time - tini));
						r2_x_p = xcoord(i) - 2.0 / (1.0 + mu) * rorbit_dynamic_p * cos(omegaorbit_p * (current_time - tini));
						sq_r2_p = r2_x_p * r2_x_p + r2_y_p * r2_y_p + r2_z_p * r2_z_p;
						r1_y_p = ycoord(j) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic_p * sin(omegaorbit_p * (current_time - tini));
						r1_x_p = xcoord(i) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic_p * cos(omegaorbit_p * (current_time - tini));
						sq_r1_p = r1_x_p * r1_x_p + r1_y_p * r1_y_p + r1_z_p * r1_z_p;
						Tbar_p = -(sq_r1_p * exp(-((sqrt(sq_r1_p) - rdonut) * (sqrt(sq_r1_p) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * sq_r2_p * exp(-((sqrt(sq_r2_p) - rdonut) * (sqrt(sq_r2_p) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
						gammax_off = gammax * 0.5 * (1.0 + tanh(wd * (sq_r1_p / (xf * xf) - 1.0))) * 0.5 * (1.0 + tanh(wd * (sq_r2_p / (xf * xf) - 1.0)));
						KXXf_p = 0.5 * betax + gammax_off * Xphi_p;
						KXf_p = (-0.5 * sigmax) + 0.5 * betax * Xphi_p + 0.5 * gammax_off * (Xphi_p * Xphi_p);
						d_Psisf_o0_t0_m0_l0 = 0.0;
						d_phi_o0_t0_m0_l0 = vector(Psi_p, i, j, k);
						d_Psi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * KXXf_p * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p) * Tbar_p / 4.0;
						m_Psi_o0_t0_l0 = (-2.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * (vector(Phid_x_p, i, j, k) * vector(Phid_x_p, i, j, k))) * d_Psi_o0_t0_m0_l0;
						m_Psi_o0_t1_l0 = (-4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_y_p, i, j, k) * vector(Phid_x_p, i, j, k)) * d_Psi_o0_t1_m0_l0;
						m_Psi_o0_t2_l0 = (-4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_z_p, i, j, k) * vector(Phid_x_p, i, j, k)) * d_Psi_o0_t2_m0_l0;
						m_Psi_o0_t3_l0 = 4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_x_p, i, j, k) * vector(Psi_p, i, j, k) * d_Phid_x_o0_t0_m0_l0;
						m_Psi_o0_t4_l0 = (-2.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * (vector(Phid_y_p, i, j, k) * vector(Phid_y_p, i, j, k))) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t5_l0 = (-4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_z_p, i, j, k) * vector(Phid_y_p, i, j, k)) * d_Psi_o0_t5_m0_l0;
						m_Psi_o0_t6_l0 = 4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_y_p, i, j, k) * vector(Psi_p, i, j, k) * d_Phid_y_o0_t0_m0_l0;
						m_Psi_o0_t7_l0 = (-2.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * (vector(Phid_z_p, i, j, k) * vector(Phid_z_p, i, j, k))) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t8_l0 = 4.0 / (2.0 * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - KXf_p / KXXf_p) * vector(Phid_z_p, i, j, k) * vector(Psi_p, i, j, k) * d_Phid_z_o0_t0_m0_l0;
						m_Psi_o0_t9_l0 = (-1.0 / (2.0 * KXXf_p / KXf_p * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - 1.0)) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t10_l0 = (-1.0 / (2.0 * KXXf_p / KXf_p * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - 1.0)) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t11_l0 = (-1.0 / (2.0 * KXXf_p / KXf_p * (vector(Psi_p, i, j, k) * vector(Psi_p, i, j, k)) - 1.0)) * d_Psi_o0_t0_m0_l0;
						RHS_Psisf = d_Psisf_o0_t0_m0_l0;
						RHS_phi = d_phi_o0_t0_m0_l0;
						RHS_Phid_x = d_Phid_x_o0_t0_m0_l0;
						RHS_Phid_y = d_Phid_y_o0_t0_m0_l0;
						RHS_Phid_z = d_Phid_z_o0_t0_m0_l0;
						RHS_Psi = (((((((((((m_Psi_o0_t0_l0 + m_Psi_o0_t1_l0) + m_Psi_o0_t2_l0) + m_Psi_o0_t3_l0) + m_Psi_o0_t4_l0) + m_Psi_o0_t5_l0) + m_Psi_o0_t6_l0) + m_Psi_o0_t7_l0) + m_Psi_o0_t8_l0) + m_Psi_o0_t9_l0) + m_Psi_o0_t10_l0) + m_Psi_o0_t11_l0) + d_Psi_o0_t12_m0_l0;
						n_x = 0.0;
						n_y = 0.0;
						n_z = 0.0;
						interaction_index = 0.0;
						if ((((((vector(FOV_xLower, i + 1, j, k) > 0.0) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i - 1, j, k) > 0.0) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i, j + 1, k) > 0.0) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j - 1, k) > 0.0) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j, k + 1) > 0.0) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((vector(FOV_xLower, i, j, k - 1) > 0.0) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((!equalsEq(n_x, 0.0) || !equalsEq(n_y, 0.0)) || !equalsEq(n_z, 0.0)) {
							mod_normal = sqrt((n_x * n_x + n_y * n_y) + n_z * n_z);
							n_x = n_x / mod_normal;
							n_y = n_y / mod_normal;
							n_z = n_z / mod_normal;
						}
						if (equalsEq(interaction_index, 1.0)) {
							i_d_Phid_y_o0_t0_m0_l0 = D1CDO4_i(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t0_m0_l0 = D1CDO4_i(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t1_m0_l0 = D1CDO4_j(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t0_m0_l0 = D1CDO4_i(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t1_m0_l0 = D1CDO4_j(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t2_m0_l0 = D1CDO4_k(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t0_m0_l0 = D1CDO4_i(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t1_m0_l0 = D1CDO4_j(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t2_m0_l0 = D1CDO4_k(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t3_m0_l0 = -phi_falloff * (vector(phi_p, i, j, k) - phi_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_x_o0_t3_m0_l0 = -Phid_x_falloff * (vector(Phid_x_p, i, j, k) - Phid_x_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_y_o0_t3_m0_l0 = -Phid_y_falloff * (vector(Phid_y_p, i, j, k) - Phid_y_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_z_o0_t3_m0_l0 = -Phid_z_falloff * (vector(Phid_z_p, i, j, k) - Phid_z_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psisf_o0_t3_m0_l0 = -Psisf_falloff * (vector(Psisf_p, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psi_o0_t3_m0_l0 = -Psisf_falloff * (vector(Psi_p, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_m_phi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t0_m0_l0;
							i_m_phi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t1_m0_l0;
							i_m_phi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t2_m0_l0;
							i_m_Phid_x_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t0_m0_l0;
							i_m_Phid_x_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t1_m0_l0;
							i_m_Phid_x_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t2_m0_l0;
							i_m_Phid_y_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_y_o0_t0_m0_l0;
							i_m_Phid_y_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t4_m0_l0;
							i_m_Phid_y_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t5_m0_l0;
							i_m_Phid_z_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t0_m0_l0;
							i_m_Phid_z_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t1_m0_l0;
							i_m_Phid_z_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t7_m0_l0;
							i_m_Psisf_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t0_m0_l0;
							i_m_Psisf_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t1_m0_l0;
							i_m_Psisf_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t2_m0_l0;
							i_m_Psi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_x_o0_t0_m0_l0;
							i_m_Psi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_y_o0_t0_m0_l0;
							i_m_Psi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_z_o0_t0_m0_l0;
							RHS_phi = ((i_m_phi_o0_t0_l0 + i_m_phi_o0_t1_l0) + i_m_phi_o0_t2_l0) + i_d_phi_o0_t3_m0_l0;
							RHS_Phid_x = ((i_m_Phid_x_o0_t0_l0 + i_m_Phid_x_o0_t1_l0) + i_m_Phid_x_o0_t2_l0) + i_d_Phid_x_o0_t3_m0_l0;
							RHS_Phid_y = ((i_m_Phid_y_o0_t0_l0 + i_m_Phid_y_o0_t1_l0) + i_m_Phid_y_o0_t2_l0) + i_d_Phid_y_o0_t3_m0_l0;
							RHS_Phid_z = ((i_m_Phid_z_o0_t0_l0 + i_m_Phid_z_o0_t1_l0) + i_m_Phid_z_o0_t2_l0) + i_d_Phid_z_o0_t3_m0_l0;
							RHS_Psisf = ((i_m_Psisf_o0_t0_l0 + i_m_Psisf_o0_t1_l0) + i_m_Psisf_o0_t2_l0) + i_d_Psisf_o0_t3_m0_l0;
							RHS_Psi = ((i_m_Psi_o0_t0_l0 + i_m_Psi_o0_t1_l0) + i_m_Psi_o0_t2_l0) + i_d_Psi_o0_t3_m0_l0;
						}
						if (dissipation_factor_Psisf > 0.0) {
							RHS_Psisf = RHS_Psisf + dissipation_factor_Psisf * (meshDissipation_i(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(Psisf_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_phi > 0.0) {
							RHS_phi = RHS_phi + dissipation_factor_phi * (meshDissipation_i(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(phi_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_x > 0.0) {
							RHS_Phid_x = RHS_Phid_x + dissipation_factor_Phid_x * (meshDissipation_i(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(Phid_x_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_y > 0.0) {
							RHS_Phid_y = RHS_Phid_y + dissipation_factor_Phid_y * (meshDissipation_i(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(Phid_y_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_z > 0.0) {
							RHS_Phid_z = RHS_Phid_z + dissipation_factor_Phid_z * (meshDissipation_i(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(Phid_z_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Psi > 0.0) {
							RHS_Psi = RHS_Psi + dissipation_factor_Psi * (meshDissipation_i(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(Psi_p, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						vector(rk1Psisf, i, j, k) = RK4P1_(RHS_Psisf, vector(Psisf_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk1phi, i, j, k) = RK4P1_(RHS_phi, vector(phi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk1Phid_x, i, j, k) = RK4P1_(RHS_Phid_x, vector(Phid_x_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk1Phid_y, i, j, k) = RK4P1_(RHS_Phid_y, vector(Phid_y_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk1Phid_z, i, j, k) = RK4P1_(RHS_Phid_z, vector(Phid_z_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk1Psi, i, j, k) = RK4P1_(RHS_Psi, vector(Psi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
					}
				}
			}
		}
	}
	//Fill ghosts and periodical boundaries
	time_interpolate_operator_mesh1->setStep(1);
	d_bdry_sched_advance1[ln]->fillData(current_time + simPlat_dt * 0.5, false);
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* rk1phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1phi_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
		double* rk1Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_x_id).get())->getPointer();
		double* rk1Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_y_id).get())->getPointer();
		double* rk1Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_z_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if (((i + 3 < ilast || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) && (i - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) && (j + 3 < jlast || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) && (j - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) && (k + 3 < klast || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) && (k - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)))) {
						vector(phi2, i, j, k) = vector(rk1phi, i, j, k) * vector(rk1phi, i, j, k);
						vector(dphi2, i, j, k) = vector(rk1Phid_x, i, j, k) * vector(rk1Phid_x, i, j, k) + vector(rk1Phid_y, i, j, k) * vector(rk1Phid_y, i, j, k) + vector(rk1Phid_z, i, j, k) * vector(rk1Phid_z, i, j, k);
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		//Hard region field distance variables
		double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
	
		double* rk1Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psisf_id).get())->getPointer();
		double* rk1phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1phi_id).get())->getPointer();
		double* rk1Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_x_id).get())->getPointer();
		double* rk1Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_y_id).get())->getPointer();
		double* rk1Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_z_id).get())->getPointer();
		double* rk1Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psi_id).get())->getPointer();
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((vector(FOV_xLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_xUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk1Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* rk1Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psi_id).get())->getPointer();
		double* rk1Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_x_id).get())->getPointer();
		double* rk1Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_y_id).get())->getPointer();
		double* rk1Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_z_id).get())->getPointer();
		double* rk1phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1phi_id).get())->getPointer();
		double* rk1Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psisf_id).get())->getPointer();
		double* rk2Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psisf_id).get())->getPointer();
		double* Psisf_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_p_id).get())->getPointer();
		double* rk2phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2phi_id).get())->getPointer();
		double* phi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_p_id).get())->getPointer();
		double* rk2Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_x_id).get())->getPointer();
		double* Phid_x_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_p_id).get())->getPointer();
		double* rk2Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_y_id).get())->getPointer();
		double* Phid_y_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_p_id).get())->getPointer();
		double* rk2Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_z_id).get())->getPointer();
		double* Phid_z_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_p_id).get())->getPointer();
		double* rk2Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psi_id).get())->getPointer();
		double* Psi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_p_id).get())->getPointer();
		double d_Phid_x_o0_t0_m0_l0, d_Phid_y_o0_t0_m0_l0, d_Phid_z_o0_t0_m0_l0, d_Psi_o0_t0_m0_l0, d_Psi_o0_t1_m0_l0, d_Psi_o0_t2_m0_l0, d_Psi_o0_t4_m0_l0, d_Psi_o0_t5_m0_l0, d_Psi_o0_t7_m0_l0, rk1Phiu_x, rk1Phiu_z, rk1Phiu_y, rk1Xphi, rk1r1_z, rk1r2_z, rk1rorbit_dynamic, rk1omegaorbit, rk1r2_y, rk1r2_x, rk1sq_r2, rk1r1_y, rk1r1_x, rk1sq_r1, rk1Tbar, gammax_off, rk1KXXf, rk1KXf, d_Psisf_o0_t0_m0_l0, d_phi_o0_t0_m0_l0, d_Psi_o0_t12_m0_l0, m_Psi_o0_t0_l0, m_Psi_o0_t1_l0, m_Psi_o0_t2_l0, m_Psi_o0_t3_l0, m_Psi_o0_t4_l0, m_Psi_o0_t5_l0, m_Psi_o0_t6_l0, m_Psi_o0_t7_l0, m_Psi_o0_t8_l0, m_Psi_o0_t9_l0, m_Psi_o0_t10_l0, m_Psi_o0_t11_l0, RHS_Psisf, RHS_phi, RHS_Phid_x, RHS_Phid_y, RHS_Phid_z, RHS_Psi, n_x, n_y, n_z, interaction_index, mod_normal, i_d_Phid_y_o0_t0_m0_l0, i_d_Phid_z_o0_t0_m0_l0, i_d_Phid_z_o0_t1_m0_l0, i_d_phi_o0_t0_m0_l0, i_d_phi_o0_t1_m0_l0, i_d_phi_o0_t2_m0_l0, i_d_Psisf_o0_t0_m0_l0, i_d_Psisf_o0_t1_m0_l0, i_d_Psisf_o0_t2_m0_l0, i_d_phi_o0_t3_m0_l0, i_d_Phid_x_o0_t3_m0_l0, i_d_Phid_y_o0_t3_m0_l0, i_d_Phid_z_o0_t3_m0_l0, i_d_Psisf_o0_t3_m0_l0, i_d_Psi_o0_t3_m0_l0, i_m_phi_o0_t0_l0, i_m_phi_o0_t1_l0, i_m_phi_o0_t2_l0, i_m_Phid_x_o0_t0_l0, i_m_Phid_x_o0_t1_l0, i_m_Phid_x_o0_t2_l0, i_m_Phid_y_o0_t0_l0, i_m_Phid_y_o0_t1_l0, i_m_Phid_y_o0_t2_l0, i_m_Phid_z_o0_t0_l0, i_m_Phid_z_o0_t1_l0, i_m_Phid_z_o0_t2_l0, i_m_Psisf_o0_t0_l0, i_m_Psisf_o0_t1_l0, i_m_Psisf_o0_t2_l0, i_m_Psi_o0_t0_l0, i_m_Psi_o0_t1_l0, i_m_Psi_o0_t2_l0;
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((i + 3 < ilast && i - 3 >= 0 && j + 3 < jlast && j - 3 >= 0 && k + 3 < klast && k - 3 >= 0)) {
						d_Phid_x_o0_t0_m0_l0 = D1CDO4_i(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_y_o0_t0_m0_l0 = D1CDO4_j(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_z_o0_t0_m0_l0 = D1CDO4_k(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t0_m0_l0 = D1CDO4_i(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t1_m0_l0 = D1CDO4_j(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t2_m0_l0 = D1CDO4_k(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t4_m0_l0 = D1CDO4_j(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t5_m0_l0 = D1CDO4_k(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t7_m0_l0 = D1CDO4_k(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
						rk1Phiu_x = vector(rk1Phid_x, i, j, k);
						rk1Phiu_z = vector(rk1Phid_z, i, j, k);
						rk1Phiu_y = vector(rk1Phid_y, i, j, k);
						rk1Xphi = (-vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) + rk1Phiu_x * vector(rk1Phid_x, i, j, k) + rk1Phiu_y * vector(rk1Phid_y, i, j, k) + rk1Phiu_z * vector(rk1Phid_z, i, j, k);
						rk1r1_z = zcoord(k);
						rk1r2_z = zcoord(k);
						if (lessEq(current_time, torbit)) {
							rk1rorbit_dynamic = rstart;
						}
						if ((current_time > torbit) && (current_time < (torbit + tslow))) {
							rk1rorbit_dynamic = rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0);
						}
						if (greaterEq(current_time, (torbit + tslow))) {
							rk1rorbit_dynamic = rorbit;
						}
						if (lessEq(current_time, tini)) {
							rk1omegaorbit = 0.0;
						}
						if ((current_time > tini) && (current_time < (torbit + tini))) {
							rk1omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk1rorbit_dynamic) * (2.0 * rk1rorbit_dynamic) * (2.0 * rk1rorbit_dynamic))) * 4.0 * (2.0 * torbit - (current_time - tini)) * (current_time - tini) / ((2.0 * torbit) * (2.0 * torbit));
						}
						if (greaterEq(current_time, (tini + torbit))) {
							rk1omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk1rorbit_dynamic) * (2.0 * rk1rorbit_dynamic) * (2.0 * rk1rorbit_dynamic)));
						}
						rk1r2_y = ycoord(j) - 2.0 / (1.0 + mu) * rk1rorbit_dynamic * sin(rk1omegaorbit * (current_time - tini));
						rk1r2_x = xcoord(i) - 2.0 / (1.0 + mu) * rk1rorbit_dynamic * cos(rk1omegaorbit * (current_time - tini));
						rk1sq_r2 = rk1r2_x * rk1r2_x + rk1r2_y * rk1r2_y + rk1r2_z * rk1r2_z;
						rk1r1_y = ycoord(j) + 2.0 * mu / (1.0 + mu) * rk1rorbit_dynamic * sin(rk1omegaorbit * (current_time - tini));
						rk1r1_x = xcoord(i) + 2.0 * mu / (1.0 + mu) * rk1rorbit_dynamic * cos(rk1omegaorbit * (current_time - tini));
						rk1sq_r1 = rk1r1_x * rk1r1_x + rk1r1_y * rk1r1_y + rk1r1_z * rk1r1_z;
						rk1Tbar = -(rk1sq_r1 * exp(-((sqrt(rk1sq_r1) - rdonut) * (sqrt(rk1sq_r1) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * rk1sq_r2 * exp(-((sqrt(rk1sq_r2) - rdonut) * (sqrt(rk1sq_r2) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
						gammax_off = gammax * 0.5 * (1.0 + tanh(wd * (rk1sq_r1 / (xf * xf) - 1.0))) * 0.5 * (1.0 + tanh(wd * (rk1sq_r2 / (xf * xf) - 1.0)));
						rk1KXXf = 0.5 * betax + gammax_off * rk1Xphi;
						rk1KXf = (-0.5 * sigmax) + 0.5 * betax * rk1Xphi + 0.5 * gammax_off * (rk1Xphi * rk1Xphi);
						d_Psisf_o0_t0_m0_l0 = 0.0;
						d_phi_o0_t0_m0_l0 = vector(rk1Psi, i, j, k);
						d_Psi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * rk1KXXf * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf) * rk1Tbar / 4.0;
						m_Psi_o0_t0_l0 = (-2.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * (vector(rk1Phid_x, i, j, k) * vector(rk1Phid_x, i, j, k))) * d_Psi_o0_t0_m0_l0;
						m_Psi_o0_t1_l0 = (-4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_y, i, j, k) * vector(rk1Phid_x, i, j, k)) * d_Psi_o0_t1_m0_l0;
						m_Psi_o0_t2_l0 = (-4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_z, i, j, k) * vector(rk1Phid_x, i, j, k)) * d_Psi_o0_t2_m0_l0;
						m_Psi_o0_t3_l0 = 4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_x, i, j, k) * vector(rk1Psi, i, j, k) * d_Phid_x_o0_t0_m0_l0;
						m_Psi_o0_t4_l0 = (-2.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * (vector(rk1Phid_y, i, j, k) * vector(rk1Phid_y, i, j, k))) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t5_l0 = (-4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_z, i, j, k) * vector(rk1Phid_y, i, j, k)) * d_Psi_o0_t5_m0_l0;
						m_Psi_o0_t6_l0 = 4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_y, i, j, k) * vector(rk1Psi, i, j, k) * d_Phid_y_o0_t0_m0_l0;
						m_Psi_o0_t7_l0 = (-2.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * (vector(rk1Phid_z, i, j, k) * vector(rk1Phid_z, i, j, k))) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t8_l0 = 4.0 / (2.0 * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - rk1KXf / rk1KXXf) * vector(rk1Phid_z, i, j, k) * vector(rk1Psi, i, j, k) * d_Phid_z_o0_t0_m0_l0;
						m_Psi_o0_t9_l0 = (-1.0 / (2.0 * rk1KXXf / rk1KXf * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - 1.0)) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t10_l0 = (-1.0 / (2.0 * rk1KXXf / rk1KXf * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - 1.0)) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t11_l0 = (-1.0 / (2.0 * rk1KXXf / rk1KXf * (vector(rk1Psi, i, j, k) * vector(rk1Psi, i, j, k)) - 1.0)) * d_Psi_o0_t0_m0_l0;
						RHS_Psisf = d_Psisf_o0_t0_m0_l0;
						RHS_phi = d_phi_o0_t0_m0_l0;
						RHS_Phid_x = d_Phid_x_o0_t0_m0_l0;
						RHS_Phid_y = d_Phid_y_o0_t0_m0_l0;
						RHS_Phid_z = d_Phid_z_o0_t0_m0_l0;
						RHS_Psi = (((((((((((m_Psi_o0_t0_l0 + m_Psi_o0_t1_l0) + m_Psi_o0_t2_l0) + m_Psi_o0_t3_l0) + m_Psi_o0_t4_l0) + m_Psi_o0_t5_l0) + m_Psi_o0_t6_l0) + m_Psi_o0_t7_l0) + m_Psi_o0_t8_l0) + m_Psi_o0_t9_l0) + m_Psi_o0_t10_l0) + m_Psi_o0_t11_l0) + d_Psi_o0_t12_m0_l0;
						n_x = 0.0;
						n_y = 0.0;
						n_z = 0.0;
						interaction_index = 0.0;
						if ((((((vector(FOV_xLower, i + 1, j, k) > 0.0) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i - 1, j, k) > 0.0) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i, j + 1, k) > 0.0) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j - 1, k) > 0.0) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j, k + 1) > 0.0) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((vector(FOV_xLower, i, j, k - 1) > 0.0) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((!equalsEq(n_x, 0.0) || !equalsEq(n_y, 0.0)) || !equalsEq(n_z, 0.0)) {
							mod_normal = sqrt((n_x * n_x + n_y * n_y) + n_z * n_z);
							n_x = n_x / mod_normal;
							n_y = n_y / mod_normal;
							n_z = n_z / mod_normal;
						}
						if (equalsEq(interaction_index, 1.0)) {
							i_d_Phid_y_o0_t0_m0_l0 = D1CDO4_i(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t0_m0_l0 = D1CDO4_i(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t1_m0_l0 = D1CDO4_j(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t0_m0_l0 = D1CDO4_i(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t1_m0_l0 = D1CDO4_j(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t2_m0_l0 = D1CDO4_k(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t0_m0_l0 = D1CDO4_i(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t1_m0_l0 = D1CDO4_j(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t2_m0_l0 = D1CDO4_k(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t3_m0_l0 = -phi_falloff * (vector(rk1phi, i, j, k) - phi_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_x_o0_t3_m0_l0 = -Phid_x_falloff * (vector(rk1Phid_x, i, j, k) - Phid_x_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_y_o0_t3_m0_l0 = -Phid_y_falloff * (vector(rk1Phid_y, i, j, k) - Phid_y_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_z_o0_t3_m0_l0 = -Phid_z_falloff * (vector(rk1Phid_z, i, j, k) - Phid_z_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psisf_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk1Psisf, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psi_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk1Psi, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_m_phi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t0_m0_l0;
							i_m_phi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t1_m0_l0;
							i_m_phi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t2_m0_l0;
							i_m_Phid_x_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t0_m0_l0;
							i_m_Phid_x_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t1_m0_l0;
							i_m_Phid_x_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t2_m0_l0;
							i_m_Phid_y_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_y_o0_t0_m0_l0;
							i_m_Phid_y_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t4_m0_l0;
							i_m_Phid_y_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t5_m0_l0;
							i_m_Phid_z_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t0_m0_l0;
							i_m_Phid_z_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t1_m0_l0;
							i_m_Phid_z_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t7_m0_l0;
							i_m_Psisf_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t0_m0_l0;
							i_m_Psisf_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t1_m0_l0;
							i_m_Psisf_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t2_m0_l0;
							i_m_Psi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_x_o0_t0_m0_l0;
							i_m_Psi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_y_o0_t0_m0_l0;
							i_m_Psi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_z_o0_t0_m0_l0;
							RHS_phi = ((i_m_phi_o0_t0_l0 + i_m_phi_o0_t1_l0) + i_m_phi_o0_t2_l0) + i_d_phi_o0_t3_m0_l0;
							RHS_Phid_x = ((i_m_Phid_x_o0_t0_l0 + i_m_Phid_x_o0_t1_l0) + i_m_Phid_x_o0_t2_l0) + i_d_Phid_x_o0_t3_m0_l0;
							RHS_Phid_y = ((i_m_Phid_y_o0_t0_l0 + i_m_Phid_y_o0_t1_l0) + i_m_Phid_y_o0_t2_l0) + i_d_Phid_y_o0_t3_m0_l0;
							RHS_Phid_z = ((i_m_Phid_z_o0_t0_l0 + i_m_Phid_z_o0_t1_l0) + i_m_Phid_z_o0_t2_l0) + i_d_Phid_z_o0_t3_m0_l0;
							RHS_Psisf = ((i_m_Psisf_o0_t0_l0 + i_m_Psisf_o0_t1_l0) + i_m_Psisf_o0_t2_l0) + i_d_Psisf_o0_t3_m0_l0;
							RHS_Psi = ((i_m_Psi_o0_t0_l0 + i_m_Psi_o0_t1_l0) + i_m_Psi_o0_t2_l0) + i_d_Psi_o0_t3_m0_l0;
						}
						if (dissipation_factor_Psisf > 0.0) {
							RHS_Psisf = RHS_Psisf + dissipation_factor_Psisf * (meshDissipation_i(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1Psisf, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_phi > 0.0) {
							RHS_phi = RHS_phi + dissipation_factor_phi * (meshDissipation_i(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1phi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_x > 0.0) {
							RHS_Phid_x = RHS_Phid_x + dissipation_factor_Phid_x * (meshDissipation_i(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_y > 0.0) {
							RHS_Phid_y = RHS_Phid_y + dissipation_factor_Phid_y * (meshDissipation_i(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_z > 0.0) {
							RHS_Phid_z = RHS_Phid_z + dissipation_factor_Phid_z * (meshDissipation_i(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Psi > 0.0) {
							RHS_Psi = RHS_Psi + dissipation_factor_Psi * (meshDissipation_i(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk1Psi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						vector(rk2Psisf, i, j, k) = RK4P2_(RHS_Psisf, vector(Psisf_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk2phi, i, j, k) = RK4P2_(RHS_phi, vector(phi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk2Phid_x, i, j, k) = RK4P2_(RHS_Phid_x, vector(Phid_x_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk2Phid_y, i, j, k) = RK4P2_(RHS_Phid_y, vector(Phid_y_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk2Phid_z, i, j, k) = RK4P2_(RHS_Phid_z, vector(Phid_z_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk2Psi, i, j, k) = RK4P2_(RHS_Psi, vector(Psi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
					}
				}
			}
		}
	}
	//Fill ghosts and periodical boundaries
	time_interpolate_operator_mesh1->setStep(2);
	d_bdry_sched_advance7[ln]->fillData(current_time + simPlat_dt * 0.5, false);
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* rk2phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2phi_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
		double* rk2Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_x_id).get())->getPointer();
		double* rk2Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_y_id).get())->getPointer();
		double* rk2Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_z_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if (((i + 3 < ilast || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) && (i - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) && (j + 3 < jlast || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) && (j - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) && (k + 3 < klast || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) && (k - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)))) {
						vector(phi2, i, j, k) = vector(rk2phi, i, j, k) * vector(rk2phi, i, j, k);
						vector(dphi2, i, j, k) = vector(rk2Phid_x, i, j, k) * vector(rk2Phid_x, i, j, k) + vector(rk2Phid_y, i, j, k) * vector(rk2Phid_y, i, j, k) + vector(rk2Phid_z, i, j, k) * vector(rk2Phid_z, i, j, k);
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		//Hard region field distance variables
		double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
	
		double* rk2Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psisf_id).get())->getPointer();
		double* rk2phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2phi_id).get())->getPointer();
		double* rk2Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_x_id).get())->getPointer();
		double* rk2Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_y_id).get())->getPointer();
		double* rk2Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_z_id).get())->getPointer();
		double* rk2Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psi_id).get())->getPointer();
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((vector(FOV_xLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_xUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk2Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* rk2Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psi_id).get())->getPointer();
		double* rk2Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_x_id).get())->getPointer();
		double* rk2Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_y_id).get())->getPointer();
		double* rk2Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_z_id).get())->getPointer();
		double* rk2phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2phi_id).get())->getPointer();
		double* rk2Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psisf_id).get())->getPointer();
		double* rk3Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psisf_id).get())->getPointer();
		double* Psisf_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_p_id).get())->getPointer();
		double* rk3phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3phi_id).get())->getPointer();
		double* phi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_p_id).get())->getPointer();
		double* rk3Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_x_id).get())->getPointer();
		double* Phid_x_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_p_id).get())->getPointer();
		double* rk3Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_y_id).get())->getPointer();
		double* Phid_y_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_p_id).get())->getPointer();
		double* rk3Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_z_id).get())->getPointer();
		double* Phid_z_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_p_id).get())->getPointer();
		double* rk3Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psi_id).get())->getPointer();
		double* Psi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_p_id).get())->getPointer();
		double d_Phid_x_o0_t0_m0_l0, d_Phid_y_o0_t0_m0_l0, d_Phid_z_o0_t0_m0_l0, d_Psi_o0_t0_m0_l0, d_Psi_o0_t1_m0_l0, d_Psi_o0_t2_m0_l0, d_Psi_o0_t4_m0_l0, d_Psi_o0_t5_m0_l0, d_Psi_o0_t7_m0_l0, rk2Phiu_x, rk2Phiu_z, rk2Phiu_y, rk2Xphi, rk2r1_z, rk2r2_z, rk2rorbit_dynamic, rk2omegaorbit, rk2r2_y, rk2r2_x, rk2sq_r2, rk2r1_y, rk2r1_x, rk2sq_r1, rk2Tbar, gammax_off, rk2KXXf, rk2KXf, d_Psisf_o0_t0_m0_l0, d_phi_o0_t0_m0_l0, d_Psi_o0_t12_m0_l0, m_Psi_o0_t0_l0, m_Psi_o0_t1_l0, m_Psi_o0_t2_l0, m_Psi_o0_t3_l0, m_Psi_o0_t4_l0, m_Psi_o0_t5_l0, m_Psi_o0_t6_l0, m_Psi_o0_t7_l0, m_Psi_o0_t8_l0, m_Psi_o0_t9_l0, m_Psi_o0_t10_l0, m_Psi_o0_t11_l0, RHS_Psisf, RHS_phi, RHS_Phid_x, RHS_Phid_y, RHS_Phid_z, RHS_Psi, n_x, n_y, n_z, interaction_index, mod_normal, i_d_Phid_y_o0_t0_m0_l0, i_d_Phid_z_o0_t0_m0_l0, i_d_Phid_z_o0_t1_m0_l0, i_d_phi_o0_t0_m0_l0, i_d_phi_o0_t1_m0_l0, i_d_phi_o0_t2_m0_l0, i_d_Psisf_o0_t0_m0_l0, i_d_Psisf_o0_t1_m0_l0, i_d_Psisf_o0_t2_m0_l0, i_d_phi_o0_t3_m0_l0, i_d_Phid_x_o0_t3_m0_l0, i_d_Phid_y_o0_t3_m0_l0, i_d_Phid_z_o0_t3_m0_l0, i_d_Psisf_o0_t3_m0_l0, i_d_Psi_o0_t3_m0_l0, i_m_phi_o0_t0_l0, i_m_phi_o0_t1_l0, i_m_phi_o0_t2_l0, i_m_Phid_x_o0_t0_l0, i_m_Phid_x_o0_t1_l0, i_m_Phid_x_o0_t2_l0, i_m_Phid_y_o0_t0_l0, i_m_Phid_y_o0_t1_l0, i_m_Phid_y_o0_t2_l0, i_m_Phid_z_o0_t0_l0, i_m_Phid_z_o0_t1_l0, i_m_Phid_z_o0_t2_l0, i_m_Psisf_o0_t0_l0, i_m_Psisf_o0_t1_l0, i_m_Psisf_o0_t2_l0, i_m_Psi_o0_t0_l0, i_m_Psi_o0_t1_l0, i_m_Psi_o0_t2_l0;
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((i + 3 < ilast && i - 3 >= 0 && j + 3 < jlast && j - 3 >= 0 && k + 3 < klast && k - 3 >= 0)) {
						d_Phid_x_o0_t0_m0_l0 = D1CDO4_i(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_y_o0_t0_m0_l0 = D1CDO4_j(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_z_o0_t0_m0_l0 = D1CDO4_k(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t0_m0_l0 = D1CDO4_i(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t1_m0_l0 = D1CDO4_j(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t2_m0_l0 = D1CDO4_k(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t4_m0_l0 = D1CDO4_j(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t5_m0_l0 = D1CDO4_k(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t7_m0_l0 = D1CDO4_k(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
						rk2Phiu_x = vector(rk2Phid_x, i, j, k);
						rk2Phiu_z = vector(rk2Phid_z, i, j, k);
						rk2Phiu_y = vector(rk2Phid_y, i, j, k);
						rk2Xphi = (-vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) + rk2Phiu_x * vector(rk2Phid_x, i, j, k) + rk2Phiu_y * vector(rk2Phid_y, i, j, k) + rk2Phiu_z * vector(rk2Phid_z, i, j, k);
						rk2r1_z = zcoord(k);
						rk2r2_z = zcoord(k);
						if (lessEq(current_time, torbit)) {
							rk2rorbit_dynamic = rstart;
						}
						if ((current_time > torbit) && (current_time < (torbit + tslow))) {
							rk2rorbit_dynamic = rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0);
						}
						if (greaterEq(current_time, (torbit + tslow))) {
							rk2rorbit_dynamic = rorbit;
						}
						if (lessEq(current_time, tini)) {
							rk2omegaorbit = 0.0;
						}
						if ((current_time > tini) && (current_time < (torbit + tini))) {
							rk2omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk2rorbit_dynamic) * (2.0 * rk2rorbit_dynamic) * (2.0 * rk2rorbit_dynamic))) * 4.0 * (2.0 * torbit - (current_time - tini)) * (current_time - tini) / ((2.0 * torbit) * (2.0 * torbit));
						}
						if (greaterEq(current_time, (tini + torbit))) {
							rk2omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk2rorbit_dynamic) * (2.0 * rk2rorbit_dynamic) * (2.0 * rk2rorbit_dynamic)));
						}
						rk2r2_y = ycoord(j) - 2.0 / (1.0 + mu) * rk2rorbit_dynamic * sin(rk2omegaorbit * (current_time - tini));
						rk2r2_x = xcoord(i) - 2.0 / (1.0 + mu) * rk2rorbit_dynamic * cos(rk2omegaorbit * (current_time - tini));
						rk2sq_r2 = rk2r2_x * rk2r2_x + rk2r2_y * rk2r2_y + rk2r2_z * rk2r2_z;
						rk2r1_y = ycoord(j) + 2.0 * mu / (1.0 + mu) * rk2rorbit_dynamic * sin(rk2omegaorbit * (current_time - tini));
						rk2r1_x = xcoord(i) + 2.0 * mu / (1.0 + mu) * rk2rorbit_dynamic * cos(rk2omegaorbit * (current_time - tini));
						rk2sq_r1 = rk2r1_x * rk2r1_x + rk2r1_y * rk2r1_y + rk2r1_z * rk2r1_z;
						rk2Tbar = -(rk2sq_r1 * exp(-((sqrt(rk2sq_r1) - rdonut) * (sqrt(rk2sq_r1) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * rk2sq_r2 * exp(-((sqrt(rk2sq_r2) - rdonut) * (sqrt(rk2sq_r2) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
						gammax_off = gammax * 0.5 * (1.0 + tanh(wd * (rk2sq_r1 / (xf * xf) - 1.0))) * 0.5 * (1.0 + tanh(wd * (rk2sq_r2 / (xf * xf) - 1.0)));
						rk2KXXf = 0.5 * betax + gammax_off * rk2Xphi;
						rk2KXf = (-0.5 * sigmax) + 0.5 * betax * rk2Xphi + 0.5 * gammax_off * (rk2Xphi * rk2Xphi);
						d_Psisf_o0_t0_m0_l0 = 0.0;
						d_phi_o0_t0_m0_l0 = vector(rk2Psi, i, j, k);
						d_Psi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * rk2KXXf * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf) * rk2Tbar / 4.0;
						m_Psi_o0_t0_l0 = (-2.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * (vector(rk2Phid_x, i, j, k) * vector(rk2Phid_x, i, j, k))) * d_Psi_o0_t0_m0_l0;
						m_Psi_o0_t1_l0 = (-4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_y, i, j, k) * vector(rk2Phid_x, i, j, k)) * d_Psi_o0_t1_m0_l0;
						m_Psi_o0_t2_l0 = (-4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_z, i, j, k) * vector(rk2Phid_x, i, j, k)) * d_Psi_o0_t2_m0_l0;
						m_Psi_o0_t3_l0 = 4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_x, i, j, k) * vector(rk2Psi, i, j, k) * d_Phid_x_o0_t0_m0_l0;
						m_Psi_o0_t4_l0 = (-2.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * (vector(rk2Phid_y, i, j, k) * vector(rk2Phid_y, i, j, k))) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t5_l0 = (-4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_z, i, j, k) * vector(rk2Phid_y, i, j, k)) * d_Psi_o0_t5_m0_l0;
						m_Psi_o0_t6_l0 = 4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_y, i, j, k) * vector(rk2Psi, i, j, k) * d_Phid_y_o0_t0_m0_l0;
						m_Psi_o0_t7_l0 = (-2.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * (vector(rk2Phid_z, i, j, k) * vector(rk2Phid_z, i, j, k))) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t8_l0 = 4.0 / (2.0 * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - rk2KXf / rk2KXXf) * vector(rk2Phid_z, i, j, k) * vector(rk2Psi, i, j, k) * d_Phid_z_o0_t0_m0_l0;
						m_Psi_o0_t9_l0 = (-1.0 / (2.0 * rk2KXXf / rk2KXf * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - 1.0)) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t10_l0 = (-1.0 / (2.0 * rk2KXXf / rk2KXf * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - 1.0)) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t11_l0 = (-1.0 / (2.0 * rk2KXXf / rk2KXf * (vector(rk2Psi, i, j, k) * vector(rk2Psi, i, j, k)) - 1.0)) * d_Psi_o0_t0_m0_l0;
						RHS_Psisf = d_Psisf_o0_t0_m0_l0;
						RHS_phi = d_phi_o0_t0_m0_l0;
						RHS_Phid_x = d_Phid_x_o0_t0_m0_l0;
						RHS_Phid_y = d_Phid_y_o0_t0_m0_l0;
						RHS_Phid_z = d_Phid_z_o0_t0_m0_l0;
						RHS_Psi = (((((((((((m_Psi_o0_t0_l0 + m_Psi_o0_t1_l0) + m_Psi_o0_t2_l0) + m_Psi_o0_t3_l0) + m_Psi_o0_t4_l0) + m_Psi_o0_t5_l0) + m_Psi_o0_t6_l0) + m_Psi_o0_t7_l0) + m_Psi_o0_t8_l0) + m_Psi_o0_t9_l0) + m_Psi_o0_t10_l0) + m_Psi_o0_t11_l0) + d_Psi_o0_t12_m0_l0;
						n_x = 0.0;
						n_y = 0.0;
						n_z = 0.0;
						interaction_index = 0.0;
						if ((((((vector(FOV_xLower, i + 1, j, k) > 0.0) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i - 1, j, k) > 0.0) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i, j + 1, k) > 0.0) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j - 1, k) > 0.0) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j, k + 1) > 0.0) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((vector(FOV_xLower, i, j, k - 1) > 0.0) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((!equalsEq(n_x, 0.0) || !equalsEq(n_y, 0.0)) || !equalsEq(n_z, 0.0)) {
							mod_normal = sqrt((n_x * n_x + n_y * n_y) + n_z * n_z);
							n_x = n_x / mod_normal;
							n_y = n_y / mod_normal;
							n_z = n_z / mod_normal;
						}
						if (equalsEq(interaction_index, 1.0)) {
							i_d_Phid_y_o0_t0_m0_l0 = D1CDO4_i(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t0_m0_l0 = D1CDO4_i(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t1_m0_l0 = D1CDO4_j(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t0_m0_l0 = D1CDO4_i(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t1_m0_l0 = D1CDO4_j(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t2_m0_l0 = D1CDO4_k(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t0_m0_l0 = D1CDO4_i(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t1_m0_l0 = D1CDO4_j(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t2_m0_l0 = D1CDO4_k(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t3_m0_l0 = -phi_falloff * (vector(rk2phi, i, j, k) - phi_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_x_o0_t3_m0_l0 = -Phid_x_falloff * (vector(rk2Phid_x, i, j, k) - Phid_x_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_y_o0_t3_m0_l0 = -Phid_y_falloff * (vector(rk2Phid_y, i, j, k) - Phid_y_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_z_o0_t3_m0_l0 = -Phid_z_falloff * (vector(rk2Phid_z, i, j, k) - Phid_z_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psisf_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk2Psisf, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psi_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk2Psi, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_m_phi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t0_m0_l0;
							i_m_phi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t1_m0_l0;
							i_m_phi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t2_m0_l0;
							i_m_Phid_x_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t0_m0_l0;
							i_m_Phid_x_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t1_m0_l0;
							i_m_Phid_x_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t2_m0_l0;
							i_m_Phid_y_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_y_o0_t0_m0_l0;
							i_m_Phid_y_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t4_m0_l0;
							i_m_Phid_y_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t5_m0_l0;
							i_m_Phid_z_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t0_m0_l0;
							i_m_Phid_z_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t1_m0_l0;
							i_m_Phid_z_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t7_m0_l0;
							i_m_Psisf_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t0_m0_l0;
							i_m_Psisf_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t1_m0_l0;
							i_m_Psisf_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t2_m0_l0;
							i_m_Psi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_x_o0_t0_m0_l0;
							i_m_Psi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_y_o0_t0_m0_l0;
							i_m_Psi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_z_o0_t0_m0_l0;
							RHS_phi = ((i_m_phi_o0_t0_l0 + i_m_phi_o0_t1_l0) + i_m_phi_o0_t2_l0) + i_d_phi_o0_t3_m0_l0;
							RHS_Phid_x = ((i_m_Phid_x_o0_t0_l0 + i_m_Phid_x_o0_t1_l0) + i_m_Phid_x_o0_t2_l0) + i_d_Phid_x_o0_t3_m0_l0;
							RHS_Phid_y = ((i_m_Phid_y_o0_t0_l0 + i_m_Phid_y_o0_t1_l0) + i_m_Phid_y_o0_t2_l0) + i_d_Phid_y_o0_t3_m0_l0;
							RHS_Phid_z = ((i_m_Phid_z_o0_t0_l0 + i_m_Phid_z_o0_t1_l0) + i_m_Phid_z_o0_t2_l0) + i_d_Phid_z_o0_t3_m0_l0;
							RHS_Psisf = ((i_m_Psisf_o0_t0_l0 + i_m_Psisf_o0_t1_l0) + i_m_Psisf_o0_t2_l0) + i_d_Psisf_o0_t3_m0_l0;
							RHS_Psi = ((i_m_Psi_o0_t0_l0 + i_m_Psi_o0_t1_l0) + i_m_Psi_o0_t2_l0) + i_d_Psi_o0_t3_m0_l0;
						}
						if (dissipation_factor_Psisf > 0.0) {
							RHS_Psisf = RHS_Psisf + dissipation_factor_Psisf * (meshDissipation_i(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2Psisf, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_phi > 0.0) {
							RHS_phi = RHS_phi + dissipation_factor_phi * (meshDissipation_i(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2phi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_x > 0.0) {
							RHS_Phid_x = RHS_Phid_x + dissipation_factor_Phid_x * (meshDissipation_i(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_y > 0.0) {
							RHS_Phid_y = RHS_Phid_y + dissipation_factor_Phid_y * (meshDissipation_i(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_z > 0.0) {
							RHS_Phid_z = RHS_Phid_z + dissipation_factor_Phid_z * (meshDissipation_i(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Psi > 0.0) {
							RHS_Psi = RHS_Psi + dissipation_factor_Psi * (meshDissipation_i(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk2Psi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						vector(rk3Psisf, i, j, k) = RK4P3_(RHS_Psisf, vector(Psisf_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk3phi, i, j, k) = RK4P3_(RHS_phi, vector(phi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk3Phid_x, i, j, k) = RK4P3_(RHS_Phid_x, vector(Phid_x_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk3Phid_y, i, j, k) = RK4P3_(RHS_Phid_y, vector(Phid_y_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk3Phid_z, i, j, k) = RK4P3_(RHS_Phid_z, vector(Phid_z_p, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(rk3Psi, i, j, k) = RK4P3_(RHS_Psi, vector(Psi_p, i, j, k), dx, simPlat_dt, ilast, jlast);
					}
				}
			}
		}
	}
	//Fill ghosts and periodical boundaries
	time_interpolate_operator_mesh1->setStep(3);
	d_bdry_sched_advance13[ln]->fillData(current_time + simPlat_dt, false);
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* rk3phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3phi_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
		double* rk3Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_x_id).get())->getPointer();
		double* rk3Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_y_id).get())->getPointer();
		double* rk3Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_z_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if (((i + 3 < ilast || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) && (i - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) && (j + 3 < jlast || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) && (j - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) && (k + 3 < klast || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) && (k - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)))) {
						vector(phi2, i, j, k) = vector(rk3phi, i, j, k) * vector(rk3phi, i, j, k);
						vector(dphi2, i, j, k) = vector(rk3Phid_x, i, j, k) * vector(rk3Phid_x, i, j, k) + vector(rk3Phid_y, i, j, k) * vector(rk3Phid_y, i, j, k) + vector(rk3Phid_z, i, j, k) * vector(rk3Phid_z, i, j, k);
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		//Hard region field distance variables
		double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
	
		double* rk3Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psisf_id).get())->getPointer();
		double* rk3phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3phi_id).get())->getPointer();
		double* rk3Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_x_id).get())->getPointer();
		double* rk3Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_y_id).get())->getPointer();
		double* rk3Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_z_id).get())->getPointer();
		double* rk3Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psi_id).get())->getPointer();
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((vector(FOV_xLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_xUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, rk3Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* rk3Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psi_id).get())->getPointer();
		double* rk3Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_x_id).get())->getPointer();
		double* rk3Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_y_id).get())->getPointer();
		double* rk3Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Phid_z_id).get())->getPointer();
		double* rk3phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3phi_id).get())->getPointer();
		double* rk3Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk3Psisf_id).get())->getPointer();
		double* Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_id).get())->getPointer();
		double* Psisf_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_p_id).get())->getPointer();
		double* rk1Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psisf_id).get())->getPointer();
		double* rk2Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psisf_id).get())->getPointer();
		double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
		double* phi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_p_id).get())->getPointer();
		double* rk1phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1phi_id).get())->getPointer();
		double* rk2phi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2phi_id).get())->getPointer();
		double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
		double* Phid_x_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_p_id).get())->getPointer();
		double* rk1Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_x_id).get())->getPointer();
		double* rk2Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_x_id).get())->getPointer();
		double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
		double* Phid_y_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_p_id).get())->getPointer();
		double* rk1Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_y_id).get())->getPointer();
		double* rk2Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_y_id).get())->getPointer();
		double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
		double* Phid_z_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_p_id).get())->getPointer();
		double* rk1Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Phid_z_id).get())->getPointer();
		double* rk2Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Phid_z_id).get())->getPointer();
		double* Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_id).get())->getPointer();
		double* Psi_p = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_p_id).get())->getPointer();
		double* rk1Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk1Psi_id).get())->getPointer();
		double* rk2Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_rk2Psi_id).get())->getPointer();
		double d_Phid_x_o0_t0_m0_l0, d_Phid_y_o0_t0_m0_l0, d_Phid_z_o0_t0_m0_l0, d_Psi_o0_t0_m0_l0, d_Psi_o0_t1_m0_l0, d_Psi_o0_t2_m0_l0, d_Psi_o0_t4_m0_l0, d_Psi_o0_t5_m0_l0, d_Psi_o0_t7_m0_l0, rk3Phiu_x, rk3Phiu_z, rk3Phiu_y, rk3Xphi, rk3r1_z, rk3r2_z, rk3rorbit_dynamic, rk3omegaorbit, rk3r2_y, rk3r2_x, rk3sq_r2, rk3r1_y, rk3r1_x, rk3sq_r1, rk3Tbar, gammax_off, rk3KXXf, rk3KXf, d_Psisf_o0_t0_m0_l0, d_phi_o0_t0_m0_l0, d_Psi_o0_t12_m0_l0, m_Psi_o0_t0_l0, m_Psi_o0_t1_l0, m_Psi_o0_t2_l0, m_Psi_o0_t3_l0, m_Psi_o0_t4_l0, m_Psi_o0_t5_l0, m_Psi_o0_t6_l0, m_Psi_o0_t7_l0, m_Psi_o0_t8_l0, m_Psi_o0_t9_l0, m_Psi_o0_t10_l0, m_Psi_o0_t11_l0, RHS_Psisf, RHS_phi, RHS_Phid_x, RHS_Phid_y, RHS_Phid_z, RHS_Psi, n_x, n_y, n_z, interaction_index, mod_normal, i_d_Phid_y_o0_t0_m0_l0, i_d_Phid_z_o0_t0_m0_l0, i_d_Phid_z_o0_t1_m0_l0, i_d_phi_o0_t0_m0_l0, i_d_phi_o0_t1_m0_l0, i_d_phi_o0_t2_m0_l0, i_d_Psisf_o0_t0_m0_l0, i_d_Psisf_o0_t1_m0_l0, i_d_Psisf_o0_t2_m0_l0, i_d_phi_o0_t3_m0_l0, i_d_Phid_x_o0_t3_m0_l0, i_d_Phid_y_o0_t3_m0_l0, i_d_Phid_z_o0_t3_m0_l0, i_d_Psisf_o0_t3_m0_l0, i_d_Psi_o0_t3_m0_l0, i_m_phi_o0_t0_l0, i_m_phi_o0_t1_l0, i_m_phi_o0_t2_l0, i_m_Phid_x_o0_t0_l0, i_m_Phid_x_o0_t1_l0, i_m_Phid_x_o0_t2_l0, i_m_Phid_y_o0_t0_l0, i_m_Phid_y_o0_t1_l0, i_m_Phid_y_o0_t2_l0, i_m_Phid_z_o0_t0_l0, i_m_Phid_z_o0_t1_l0, i_m_Phid_z_o0_t2_l0, i_m_Psisf_o0_t0_l0, i_m_Psisf_o0_t1_l0, i_m_Psisf_o0_t2_l0, i_m_Psi_o0_t0_l0, i_m_Psi_o0_t1_l0, i_m_Psi_o0_t2_l0;
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((i + 3 < ilast && i - 3 >= 0 && j + 3 < jlast && j - 3 >= 0 && k + 3 < klast && k - 3 >= 0)) {
						d_Phid_x_o0_t0_m0_l0 = D1CDO4_i(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_y_o0_t0_m0_l0 = D1CDO4_j(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Phid_z_o0_t0_m0_l0 = D1CDO4_k(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t0_m0_l0 = D1CDO4_i(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t1_m0_l0 = D1CDO4_j(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t2_m0_l0 = D1CDO4_k(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t4_m0_l0 = D1CDO4_j(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t5_m0_l0 = D1CDO4_k(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
						d_Psi_o0_t7_m0_l0 = D1CDO4_k(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
						rk3Phiu_x = vector(rk3Phid_x, i, j, k);
						rk3Phiu_z = vector(rk3Phid_z, i, j, k);
						rk3Phiu_y = vector(rk3Phid_y, i, j, k);
						rk3Xphi = (-vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) + rk3Phiu_x * vector(rk3Phid_x, i, j, k) + rk3Phiu_y * vector(rk3Phid_y, i, j, k) + rk3Phiu_z * vector(rk3Phid_z, i, j, k);
						rk3r1_z = zcoord(k);
						rk3r2_z = zcoord(k);
						if (lessEq(current_time, torbit)) {
							rk3rorbit_dynamic = rstart;
						}
						if ((current_time > torbit) && (current_time < (torbit + tslow))) {
							rk3rorbit_dynamic = rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0);
						}
						if (greaterEq(current_time, (torbit + tslow))) {
							rk3rorbit_dynamic = rorbit;
						}
						if (lessEq(current_time, tini)) {
							rk3omegaorbit = 0.0;
						}
						if ((current_time > tini) && (current_time < (torbit + tini))) {
							rk3omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk3rorbit_dynamic) * (2.0 * rk3rorbit_dynamic) * (2.0 * rk3rorbit_dynamic))) * 4.0 * (2.0 * torbit - (current_time - tini)) * (current_time - tini) / ((2.0 * torbit) * (2.0 * torbit));
						}
						if (greaterEq(current_time, (tini + torbit))) {
							rk3omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rk3rorbit_dynamic) * (2.0 * rk3rorbit_dynamic) * (2.0 * rk3rorbit_dynamic)));
						}
						rk3r2_y = ycoord(j) - 2.0 / (1.0 + mu) * rk3rorbit_dynamic * sin(rk3omegaorbit * (current_time - tini));
						rk3r2_x = xcoord(i) - 2.0 / (1.0 + mu) * rk3rorbit_dynamic * cos(rk3omegaorbit * (current_time - tini));
						rk3sq_r2 = rk3r2_x * rk3r2_x + rk3r2_y * rk3r2_y + rk3r2_z * rk3r2_z;
						rk3r1_y = ycoord(j) + 2.0 * mu / (1.0 + mu) * rk3rorbit_dynamic * sin(rk3omegaorbit * (current_time - tini));
						rk3r1_x = xcoord(i) + 2.0 * mu / (1.0 + mu) * rk3rorbit_dynamic * cos(rk3omegaorbit * (current_time - tini));
						rk3sq_r1 = rk3r1_x * rk3r1_x + rk3r1_y * rk3r1_y + rk3r1_z * rk3r1_z;
						rk3Tbar = -(rk3sq_r1 * exp(-((sqrt(rk3sq_r1) - rdonut) * (sqrt(rk3sq_r1) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * rk3sq_r2 * exp(-((sqrt(rk3sq_r2) - rdonut) * (sqrt(rk3sq_r2) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
						gammax_off = gammax * 0.5 * (1.0 + tanh(wd * (rk3sq_r1 / (xf * xf) - 1.0))) * 0.5 * (1.0 + tanh(wd * (rk3sq_r2 / (xf * xf) - 1.0)));
						rk3KXXf = 0.5 * betax + gammax_off * rk3Xphi;
						rk3KXf = (-0.5 * sigmax) + 0.5 * betax * rk3Xphi + 0.5 * gammax_off * (rk3Xphi * rk3Xphi);
						d_Psisf_o0_t0_m0_l0 = 0.0;
						d_phi_o0_t0_m0_l0 = vector(rk3Psi, i, j, k);
						d_Psi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * rk3KXXf * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf) * rk3Tbar / 4.0;
						m_Psi_o0_t0_l0 = (-2.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * (vector(rk3Phid_x, i, j, k) * vector(rk3Phid_x, i, j, k))) * d_Psi_o0_t0_m0_l0;
						m_Psi_o0_t1_l0 = (-4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_y, i, j, k) * vector(rk3Phid_x, i, j, k)) * d_Psi_o0_t1_m0_l0;
						m_Psi_o0_t2_l0 = (-4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_z, i, j, k) * vector(rk3Phid_x, i, j, k)) * d_Psi_o0_t2_m0_l0;
						m_Psi_o0_t3_l0 = 4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_x, i, j, k) * vector(rk3Psi, i, j, k) * d_Phid_x_o0_t0_m0_l0;
						m_Psi_o0_t4_l0 = (-2.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * (vector(rk3Phid_y, i, j, k) * vector(rk3Phid_y, i, j, k))) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t5_l0 = (-4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_z, i, j, k) * vector(rk3Phid_y, i, j, k)) * d_Psi_o0_t5_m0_l0;
						m_Psi_o0_t6_l0 = 4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_y, i, j, k) * vector(rk3Psi, i, j, k) * d_Phid_y_o0_t0_m0_l0;
						m_Psi_o0_t7_l0 = (-2.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * (vector(rk3Phid_z, i, j, k) * vector(rk3Phid_z, i, j, k))) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t8_l0 = 4.0 / (2.0 * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - rk3KXf / rk3KXXf) * vector(rk3Phid_z, i, j, k) * vector(rk3Psi, i, j, k) * d_Phid_z_o0_t0_m0_l0;
						m_Psi_o0_t9_l0 = (-1.0 / (2.0 * rk3KXXf / rk3KXf * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - 1.0)) * d_Psi_o0_t7_m0_l0;
						m_Psi_o0_t10_l0 = (-1.0 / (2.0 * rk3KXXf / rk3KXf * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - 1.0)) * d_Psi_o0_t4_m0_l0;
						m_Psi_o0_t11_l0 = (-1.0 / (2.0 * rk3KXXf / rk3KXf * (vector(rk3Psi, i, j, k) * vector(rk3Psi, i, j, k)) - 1.0)) * d_Psi_o0_t0_m0_l0;
						RHS_Psisf = d_Psisf_o0_t0_m0_l0;
						RHS_phi = d_phi_o0_t0_m0_l0;
						RHS_Phid_x = d_Phid_x_o0_t0_m0_l0;
						RHS_Phid_y = d_Phid_y_o0_t0_m0_l0;
						RHS_Phid_z = d_Phid_z_o0_t0_m0_l0;
						RHS_Psi = (((((((((((m_Psi_o0_t0_l0 + m_Psi_o0_t1_l0) + m_Psi_o0_t2_l0) + m_Psi_o0_t3_l0) + m_Psi_o0_t4_l0) + m_Psi_o0_t5_l0) + m_Psi_o0_t6_l0) + m_Psi_o0_t7_l0) + m_Psi_o0_t8_l0) + m_Psi_o0_t9_l0) + m_Psi_o0_t10_l0) + m_Psi_o0_t11_l0) + d_Psi_o0_t12_m0_l0;
						n_x = 0.0;
						n_y = 0.0;
						n_z = 0.0;
						interaction_index = 0.0;
						if ((((((vector(FOV_xLower, i + 1, j, k) > 0.0) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i - 1, j, k) > 0.0) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((vector(FOV_xLower, i, j + 1, k) > 0.0) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j - 1, k) > 0.0) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((vector(FOV_xLower, i, j, k + 1) > 0.0) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((vector(FOV_xLower, i, j, k - 1) > 0.0) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k) > 0.0)) || (vector(FOV_yLower, i + 1, j, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k) > 0.0)) || (vector(FOV_zLower, i + 1, j, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x + 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k) > 0.0)) || (vector(FOV_yLower, i - 1, j, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k) > 0.0)) || (vector(FOV_zLower, i - 1, j, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k) > 0.0)) {
							interaction_index = 1.0;
							n_x = n_x - 1.0 / dx[0];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k) > 0.0)) || (vector(FOV_yLower, i, j + 1, k) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k) > 0.0)) || (vector(FOV_zLower, i, j + 1, k) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y + 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k) > 0.0)) || (vector(FOV_yLower, i, j - 1, k) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k) > 0.0)) || (vector(FOV_zLower, i, j - 1, k) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k) > 0.0)) {
							interaction_index = 1.0;
							n_y = n_y - 1.0 / dx[1];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k + 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k + 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k + 1) > 0.0)) || (vector(FOV_xLower, i, j, k + 1) > 0.0)) || (vector(FOV_xUpper, i, j, k + 1) > 0.0)) || (vector(FOV_yLower, i, j, k + 1) > 0.0)) || (vector(FOV_yUpper, i, j, k + 1) > 0.0)) || (vector(FOV_zLower, i, j, k + 1) > 0.0)) || (vector(FOV_zUpper, i, j, k + 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z + 1.0 / dx[2];
						}
						if ((((((((((((((((((((((((((((((((((((((((((((((((((((((vector(FOV_xLower, i + 1, j + 1, k - 1) > 0.0) || (vector(FOV_xUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i + 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i + 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zLower, i - 1, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i - 1, j, k - 1) > 0.0)) || (vector(FOV_xLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j + 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j + 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zLower, i, j - 1, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j - 1, k - 1) > 0.0)) || (vector(FOV_xLower, i, j, k - 1) > 0.0)) || (vector(FOV_xUpper, i, j, k - 1) > 0.0)) || (vector(FOV_yLower, i, j, k - 1) > 0.0)) || (vector(FOV_yUpper, i, j, k - 1) > 0.0)) || (vector(FOV_zLower, i, j, k - 1) > 0.0)) || (vector(FOV_zUpper, i, j, k - 1) > 0.0)) {
							interaction_index = 1.0;
							n_z = n_z - 1.0 / dx[2];
						}
						if ((!equalsEq(n_x, 0.0) || !equalsEq(n_y, 0.0)) || !equalsEq(n_z, 0.0)) {
							mod_normal = sqrt((n_x * n_x + n_y * n_y) + n_z * n_z);
							n_x = n_x / mod_normal;
							n_y = n_y / mod_normal;
							n_z = n_z / mod_normal;
						}
						if (equalsEq(interaction_index, 1.0)) {
							i_d_Phid_y_o0_t0_m0_l0 = D1CDO4_i(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t0_m0_l0 = D1CDO4_i(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Phid_z_o0_t1_m0_l0 = D1CDO4_j(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t0_m0_l0 = D1CDO4_i(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t1_m0_l0 = D1CDO4_j(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t2_m0_l0 = D1CDO4_k(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t0_m0_l0 = D1CDO4_i(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t1_m0_l0 = D1CDO4_j(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_Psisf_o0_t2_m0_l0 = D1CDO4_k(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
							i_d_phi_o0_t3_m0_l0 = -phi_falloff * (vector(rk3phi, i, j, k) - phi_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_x_o0_t3_m0_l0 = -Phid_x_falloff * (vector(rk3Phid_x, i, j, k) - Phid_x_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_y_o0_t3_m0_l0 = -Phid_y_falloff * (vector(rk3Phid_y, i, j, k) - Phid_y_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Phid_z_o0_t3_m0_l0 = -Phid_z_falloff * (vector(rk3Phid_z, i, j, k) - Phid_z_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psisf_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk3Psisf, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_d_Psi_o0_t3_m0_l0 = -Psisf_falloff * (vector(rk3Psi, i, j, k) - Psisf_asymptotic) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k));
							i_m_phi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t0_m0_l0;
							i_m_phi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t1_m0_l0;
							i_m_phi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_phi_o0_t2_m0_l0;
							i_m_Phid_x_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t0_m0_l0;
							i_m_Phid_x_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t1_m0_l0;
							i_m_Phid_x_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t2_m0_l0;
							i_m_Phid_y_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_y_o0_t0_m0_l0;
							i_m_Phid_y_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t4_m0_l0;
							i_m_Phid_y_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t5_m0_l0;
							i_m_Phid_z_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t0_m0_l0;
							i_m_Phid_z_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Phid_z_o0_t1_m0_l0;
							i_m_Phid_z_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Psi_o0_t7_m0_l0;
							i_m_Psisf_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t0_m0_l0;
							i_m_Psisf_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t1_m0_l0;
							i_m_Psisf_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * i_d_Psisf_o0_t2_m0_l0;
							i_m_Psi_o0_t0_l0 = (-xcoord(i) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_x_o0_t0_m0_l0;
							i_m_Psi_o0_t1_l0 = (-ycoord(j) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_y_o0_t0_m0_l0;
							i_m_Psi_o0_t2_l0 = (-zcoord(k) / sqrt(xcoord(i) * xcoord(i) + ycoord(j) * ycoord(j) + zcoord(k) * zcoord(k))) * d_Phid_z_o0_t0_m0_l0;
							RHS_phi = ((i_m_phi_o0_t0_l0 + i_m_phi_o0_t1_l0) + i_m_phi_o0_t2_l0) + i_d_phi_o0_t3_m0_l0;
							RHS_Phid_x = ((i_m_Phid_x_o0_t0_l0 + i_m_Phid_x_o0_t1_l0) + i_m_Phid_x_o0_t2_l0) + i_d_Phid_x_o0_t3_m0_l0;
							RHS_Phid_y = ((i_m_Phid_y_o0_t0_l0 + i_m_Phid_y_o0_t1_l0) + i_m_Phid_y_o0_t2_l0) + i_d_Phid_y_o0_t3_m0_l0;
							RHS_Phid_z = ((i_m_Phid_z_o0_t0_l0 + i_m_Phid_z_o0_t1_l0) + i_m_Phid_z_o0_t2_l0) + i_d_Phid_z_o0_t3_m0_l0;
							RHS_Psisf = ((i_m_Psisf_o0_t0_l0 + i_m_Psisf_o0_t1_l0) + i_m_Psisf_o0_t2_l0) + i_d_Psisf_o0_t3_m0_l0;
							RHS_Psi = ((i_m_Psi_o0_t0_l0 + i_m_Psi_o0_t1_l0) + i_m_Psi_o0_t2_l0) + i_d_Psi_o0_t3_m0_l0;
						}
						if (dissipation_factor_Psisf > 0.0) {
							RHS_Psisf = RHS_Psisf + dissipation_factor_Psisf * (meshDissipation_i(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3Psisf, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_phi > 0.0) {
							RHS_phi = RHS_phi + dissipation_factor_phi * (meshDissipation_i(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3phi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_x > 0.0) {
							RHS_Phid_x = RHS_Phid_x + dissipation_factor_Phid_x * (meshDissipation_i(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_y > 0.0) {
							RHS_Phid_y = RHS_Phid_y + dissipation_factor_Phid_y * (meshDissipation_i(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Phid_z > 0.0) {
							RHS_Phid_z = RHS_Phid_z + dissipation_factor_Phid_z * (meshDissipation_i(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						if (dissipation_factor_Psi > 0.0) {
							RHS_Psi = RHS_Psi + dissipation_factor_Psi * (meshDissipation_i(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_j(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast) + meshDissipation_k(rk3Psi, i, j, k, dx, simPlat_dt, ilast, jlast));
						}
						vector(Psisf, i, j, k) = RK4P4_(RHS_Psisf, vector(Psisf_p, i, j, k), vector(rk1Psisf, i, j, k), vector(rk2Psisf, i, j, k), vector(rk3Psisf, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(phi, i, j, k) = RK4P4_(RHS_phi, vector(phi_p, i, j, k), vector(rk1phi, i, j, k), vector(rk2phi, i, j, k), vector(rk3phi, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(Phid_x, i, j, k) = RK4P4_(RHS_Phid_x, vector(Phid_x_p, i, j, k), vector(rk1Phid_x, i, j, k), vector(rk2Phid_x, i, j, k), vector(rk3Phid_x, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(Phid_y, i, j, k) = RK4P4_(RHS_Phid_y, vector(Phid_y_p, i, j, k), vector(rk1Phid_y, i, j, k), vector(rk2Phid_y, i, j, k), vector(rk3Phid_y, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(Phid_z, i, j, k) = RK4P4_(RHS_Phid_z, vector(Phid_z_p, i, j, k), vector(rk1Phid_z, i, j, k), vector(rk2Phid_z, i, j, k), vector(rk3Phid_z, i, j, k), dx, simPlat_dt, ilast, jlast);
						vector(Psi, i, j, k) = RK4P4_(RHS_Psi, vector(Psi_p, i, j, k), vector(rk1Psi, i, j, k), vector(rk2Psi, i, j, k), vector(rk3Psi, i, j, k), dx, simPlat_dt, ilast, jlast);
					}
				}
			}
		}
	}
	//Fill ghosts and periodical boundaries
	time_interpolate_operator_mesh1->setStep(4);
	d_bdry_sched_advance19[ln]->fillData(current_time + simPlat_dt, false);
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
		double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
		double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
		double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if (((i + 3 < ilast || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 1)) && (i - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(0, 0)) && (j + 3 < jlast || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 1)) && (j - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(1, 0)) && (k + 3 < klast || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 1)) && (k - 3 >= 0 || !patch->getPatchGeometry()->getTouchesRegularBoundary(2, 0)))) {
						vector(phi2, i, j, k) = vector(phi, i, j, k) * vector(phi, i, j, k);
						vector(dphi2, i, j, k) = vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k) + vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k) + vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k);
					}
				}
			}
		}
	}
	for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
		const std::shared_ptr<hier::Patch >& patch = *p_it;
		double* FOV_1 = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_1_id).get())->getPointer();
		double* FOV_xLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xLower_id).get())->getPointer();
		double* FOV_xUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_xUpper_id).get())->getPointer();
		double* FOV_yLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yLower_id).get())->getPointer();
		double* FOV_yUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_yUpper_id).get())->getPointer();
		double* FOV_zLower = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zLower_id).get())->getPointer();
		double* FOV_zUpper = ((pdat::NodeData<double> *) patch->getPatchData(d_FOV_zUpper_id).get())->getPointer();
	
		//Hard region field distance variables
		double* d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
		double* d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2_id).get())->getPointer();
	
		double* Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_id).get())->getPointer();
		double* phi = ((pdat::NodeData<double> *) patch->getPatchData(d_phi_id).get())->getPointer();
		double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
		double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
		double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
		double* Psi = ((pdat::NodeData<double> *) patch->getPatchData(d_Psi_id).get())->getPointer();
		double* phi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi2_id).get())->getPointer();
		double* dphi2 = ((pdat::NodeData<double> *) patch->getPatchData(d_dphi2_id).get())->getPointer();
	
		//Get the dimensions of the patch
		const hier::Index boxfirst = patch->getBox().lower();
		const hier::Index boxlast = patch->getBox().upper();
	
		//Get delta spaces into an array. dx, dy, dz.
		std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
		const double* dx  = patch_geom->getDx();
	
		//Auxiliary definitions
		int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
		int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
		int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
		for(int k = 0; k < klast; k++) {
			for(int j = 0; j < jlast; j++) {
				for(int i = 0; i < ilast; i++) {
					if ((vector(FOV_xLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_xUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_yUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zLower, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
					if ((vector(FOV_zUpper, i, j, k) > 0)) {
						//Region field extrapolations
						if ((vector(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0 || vector(d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, j, k) != 0)) {
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psisf, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_x, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_y, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Phid_z, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, Psi, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, phi2, FOV_1, dx, ilast, jlast);
							extrapolate_field(d_i_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, i, d_j_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, j, d_k_Psisf_phi_Phid_x_Phid_y_Phid_z_Psi_phi2_dphi2, k, dphi2, FOV_1, dx, ilast, jlast);
						}
					}
				}
			}
		}
	}
	if (d_refinedTimeStepping) {
		if (!hierarchy->finerLevelExists(ln) && last_step) {
			int currentLevelNumber = ln;
			while (currentLevelNumber > 0 && current_iteration[currentLevelNumber] % hierarchy->getRatioToCoarserLevel(currentLevelNumber).max() == 0) {
				d_coarsen_schedule[currentLevelNumber]->coarsenData();
				d_bdry_sched_postCoarsen[currentLevelNumber - 1]->fillData(current_time, false);
				currentLevelNumber--;
			}
		}
	} else {
		if (ln > 0) {
			d_coarsen_schedule[ln]->coarsenData();
			d_bdry_sched_postCoarsen[ln - 1]->fillData(current_time, false);
		}
	}
	

	t_step->stop();
	//Analysis
	bool calculate = false;
	if (previous_iteration < next_mesh_dump_iteration && outputCycle >= next_mesh_dump_iteration) {
		calculate = true;
	}
	for (int i = 0; i < next_slice_dump_iteration.size(); i++) {
		if (analysis_slice_dump[i] && previous_iteration < next_slice_dump_iteration[i] && outputCycle >= next_slice_dump_iteration[i]) {
			calculate = true;
		}
	}
	for (int i = 0; i < next_sphere_dump_iteration.size(); i++) {
		if (analysis_sphere_dump[i] && previous_iteration < next_sphere_dump_iteration[i] && outputCycle >= next_sphere_dump_iteration[i]) {
			calculate = true;
		}
	}
	for (int i = 0; i < next_integration_dump_iteration.size(); i++) {
		if (analysis_integration_dump[i] && previous_iteration < next_integration_dump_iteration[i] && outputCycle >= next_integration_dump_iteration[i]) {
			calculate = true;
		}
	}
	for (int i = 0; i < next_point_dump_iteration.size(); i++) {
		if (analysis_point_dump[i] && previous_iteration < next_point_dump_iteration[i] && outputCycle >= next_point_dump_iteration[i]) {
			calculate = true;
		}
	}
	if (calculate) {
		for (int ln=0; ln<=d_patch_hierarchy->getFinestLevelNumber(); ++ln ) {
			std::shared_ptr<hier::PatchLevel > level(d_patch_hierarchy->getPatchLevel(ln));
			for (hier::PatchLevel::iterator p_it(level->begin()); p_it != level->end(); ++p_it) {
				const std::shared_ptr<hier::Patch >& patch = *p_it;
		
				double* Psisf = ((pdat::NodeData<double> *) patch->getPatchData(d_Psisf_id).get())->getPointer();
				double* Phid_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_x_id).get())->getPointer();
				double* Phid_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_y_id).get())->getPointer();
				double* Phid_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Phid_z_id).get())->getPointer();
				double* d2tphi = ((pdat::NodeData<double> *) patch->getPatchData(d_d2tphi_id).get())->getPointer();
				double* Chi_t = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_t_id).get())->getPointer();
				double* Chi_x = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_x_id).get())->getPointer();
				double* Chi_y = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_y_id).get())->getPointer();
				double* Chi_z = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_z_id).get())->getPointer();
				double* phi22 = ((pdat::NodeData<double> *) patch->getPatchData(d_phi22_id).get())->getPointer();
				double* vxplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vxplus_id).get())->getPointer();
				double* vxminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vxminus_id).get())->getPointer();
				double* vyplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vyplus_id).get())->getPointer();
				double* vyminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vyminus_id).get())->getPointer();
				double* vzplus = ((pdat::NodeData<double> *) patch->getPatchData(d_vzplus_id).get())->getPointer();
				double* vzminus = ((pdat::NodeData<double> *) patch->getPatchData(d_vzminus_id).get())->getPointer();
				double* vorticity_norm = ((pdat::NodeData<double> *) patch->getPatchData(d_vorticity_norm_id).get())->getPointer();
				double* Chi_norm = ((pdat::NodeData<double> *) patch->getPatchData(d_Chi_norm_id).get())->getPointer();
				double d_Vtx_o0_t2_m0_l0, d_Vtx_o0_t1_m0_l0, d_Vtx_o0_t0_m0_l0, d_Vtx_o0_t6_m0_l0, d_Vtx_o0_t5_m0_l0, d_Vtx_o0_t4_m0_l0, d_phi22_o0_t7_m0_l0, d_Vty_o0_t6_m0_l0, d_Vty_o0_t5_m0_l0, d_phi22_o0_t10_m0_l0, d_phi22_o0_t11_m0_l0, d_Vtz_o0_t6_m0_l0, d_Vtx_o0_t7_m0_l0, m_Vtx_o0_t0_l0, m_Vtx_o0_t1_l0, m_Vtx_o0_t2_l0, m_Vtx_o0_t3_l0, m_Vtx_o0_t4_l0, m_Vtx_o0_t5_l0, m_Vtx_o0_t6_l0, Vtx, d_Vty_o0_t7_m0_l0, m_Vty_o0_t0_l0, m_Vty_o0_t1_l0, m_Vty_o0_t2_l0, m_Vty_o0_t3_l0, m_Vty_o0_t4_l0, m_Vty_o0_t5_l0, m_Vty_o0_t6_l0, Vty, d_Vtz_o0_t7_m0_l0, m_Vtz_o0_t0_l0, m_Vtz_o0_t1_l0, m_Vtz_o0_t2_l0, m_Vtz_o0_t3_l0, m_Vtz_o0_t4_l0, m_Vtz_o0_t5_l0, m_Vtz_o0_t6_l0, Vtz, m_Vxz_o0_t0_l0, m_Vxz_o0_t1_l0, m_Vxz_o0_t2_l0, m_Vxz_o0_t3_l0, m_Vxz_o0_t4_l0, m_Vxz_o0_t5_l0, m_Vxz_o0_t6_l0, m_Vxz_o0_t7_l0, Vxz, m_Vyz_o0_t0_l0, m_Vyz_o0_t1_l0, m_Vyz_o0_t2_l0, m_Vyz_o0_t3_l0, m_Vyz_o0_t4_l0, m_Vyz_o0_t5_l0, m_Vyz_o0_t6_l0, m_Vyz_o0_t7_l0, Vyz, m_Vxy_o0_t0_l0, m_Vxy_o0_t1_l0, m_Vxy_o0_t2_l0, m_Vxy_o0_t3_l0, m_Vxy_o0_t4_l0, m_Vxy_o0_t5_l0, m_Vxy_o0_t6_l0, m_Vxy_o0_t7_l0, Vxy, Phiu_x, Phiu_z, Phiu_y, Xphi, KXXf, KXf, r1_z, r2_z, rorbit_dynamic, omegaorbit, r2_y, r2_x, sq_r2, r1_y, r1_x, sq_r1, Tbar, d_phi22_o0_t0_m0_l0, d_vxplus_o0_t0_m0_l0, d_vxminus_o0_t0_m0_l0, d_vyplus_o0_t0_m0_l0, d_vyminus_o0_t0_m0_l0, d_vzplus_o0_t0_m0_l0, d_vzminus_o0_t0_m0_l0, d_vorticity_norm_o0_t0_m0_l0, d_d2tphi_o0_t12_m0_l0, d_Chi_t_o0_t0_m0_l0, d_Chi_x_o0_t0_m0_l0, d_Chi_y_o0_t0_m0_l0, d_Chi_z_o0_t0_m0_l0, d_Chi_norm_o0_t0_m0_l0, m_phi22_o0_t1_l0, m_phi22_o0_t2_l0, m_phi22_o0_t3_l0, m_phi22_o0_t4_l0, m_phi22_o0_t5_l0, m_phi22_o0_t6_l0, m_phi22_o0_t7_l0, m_phi22_o0_t8_l0, m_phi22_o0_t9_l0, m_phi22_o0_t10_l0, m_phi22_o0_t11_l0, m_phi22_o0_t12_l0, m_d2tphi_o0_t0_l0, m_d2tphi_o0_t1_l0, m_d2tphi_o0_t2_l0, m_d2tphi_o0_t3_l0, m_d2tphi_o0_t4_l0, m_d2tphi_o0_t5_l0, m_d2tphi_o0_t6_l0, m_d2tphi_o0_t7_l0, m_d2tphi_o0_t8_l0, m_d2tphi_o0_t9_l0, m_d2tphi_o0_t10_l0, m_d2tphi_o0_t11_l0;
		
				//Get the dimensions of the patch
				hier::Box pbox = patch->getBox();
				const hier::Index boxfirst = patch->getBox().lower();
				const hier::Index boxlast = patch->getBox().upper();
		
				//Get delta spaces into an array. dx, dy, dz.
				std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
				const double* dx  = patch_geom->getDx();
		
				//Auxiliary definitions
				int ilast = boxlast(0)-boxfirst(0) + 2 + 2 * d_ghost_width;
				int jlast = boxlast(1)-boxfirst(1) + 2 + 2 * d_ghost_width;
				int klast = boxlast(2)-boxfirst(2) + 2 + 2 * d_ghost_width;
			for(int k = 0; k < klast; k++) {
				for(int j = 0; j < jlast; j++) {
					for(int i = 0; i < ilast; i++) {
							if ((i + 2 < ilast && i - 2 >= 0 && j + 2 < jlast && j - 2 >= 0 && k + 2 < klast && k - 2 >= 0)) {
								d_Vtx_o0_t2_m0_l0 = D1CDO4_i(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t1_m0_l0 = D1CDO4_j(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t0_m0_l0 = D1CDO4_k(Psisf, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t6_m0_l0 = D1CDO4_i(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t5_m0_l0 = D1CDO4_j(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t4_m0_l0 = D1CDO4_k(Phid_x, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_phi22_o0_t7_m0_l0 = D1CDO4_i(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vty_o0_t6_m0_l0 = D1CDO4_j(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vty_o0_t5_m0_l0 = D1CDO4_k(Phid_y, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_phi22_o0_t10_m0_l0 = D1CDO4_i(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_phi22_o0_t11_m0_l0 = D1CDO4_j(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtz_o0_t6_m0_l0 = D1CDO4_k(Phid_z, i, j, k, dx, simPlat_dt, ilast, jlast);
								d_Vtx_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_x, i, j, k) * vector(Psisf, i, j, k);
								m_Vtx_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t0_m0_l0;
								m_Vtx_o0_t1_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t1_m0_l0;
								m_Vtx_o0_t2_l0 = (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t2_m0_l0;
								m_Vtx_o0_t3_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
								m_Vtx_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_Vtx_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t5_m0_l0;
								m_Vtx_o0_t6_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t6_m0_l0;
								Vtx = ((((((m_Vtx_o0_t0_l0 + m_Vtx_o0_t1_l0) + m_Vtx_o0_t2_l0) + m_Vtx_o0_t3_l0) + m_Vtx_o0_t4_l0) + m_Vtx_o0_t5_l0) + m_Vtx_o0_t6_l0) + d_Vtx_o0_t7_m0_l0;
								d_Vty_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_y, i, j, k) * vector(Psisf, i, j, k);
								m_Vty_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t0_m0_l0;
								m_Vty_o0_t1_l0 = (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtx_o0_t1_m0_l0;
								m_Vty_o0_t2_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t1_m0_l0;
								m_Vty_o0_t3_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t2_m0_l0;
								m_Vty_o0_t4_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t5_m0_l0;
								m_Vty_o0_t5_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vty_o0_t6_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t6_m0_l0;
								Vty = ((((((m_Vty_o0_t0_l0 + m_Vty_o0_t1_l0) + m_Vty_o0_t2_l0) + m_Vty_o0_t3_l0) + m_Vty_o0_t4_l0) + m_Vty_o0_t5_l0) + m_Vty_o0_t6_l0) + d_Vty_o0_t7_m0_l0;
								d_Vtz_o0_t7_m0_l0 = -vector(d2tphi, i, j, k) * vector(Phid_z, i, j, k) * vector(Psisf, i, j, k);
								m_Vtz_o0_t0_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vtx_o0_t0_m0_l0;
								m_Vtz_o0_t1_l0 = (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t0_m0_l0;
								m_Vtz_o0_t2_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t1_m0_l0;
								m_Vtz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t2_m0_l0;
								m_Vtz_o0_t4_l0 = (-vector(Phid_x, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_Vtz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vtz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtz_o0_t6_m0_l0;
								Vtz = ((((((m_Vtz_o0_t0_l0 + m_Vtz_o0_t1_l0) + m_Vtz_o0_t2_l0) + m_Vtz_o0_t3_l0) + m_Vtz_o0_t4_l0) + m_Vtz_o0_t5_l0) + m_Vtz_o0_t6_l0) + d_Vtz_o0_t7_m0_l0;
								m_Vxz_o0_t0_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_Vxz_o0_t1_l0 = (-vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_Vxz_o0_t2_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t5_m0_l0;
								m_Vxz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t6_m0_l0;
								m_Vxz_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
								m_Vxz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vxz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtz_o0_t6_m0_l0;
								m_Vxz_o0_t7_l0 = vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
								Vxz = ((((((m_Vxz_o0_t0_l0 + m_Vxz_o0_t1_l0) + m_Vxz_o0_t2_l0) + m_Vxz_o0_t3_l0) + m_Vxz_o0_t4_l0) + m_Vxz_o0_t5_l0) + m_Vxz_o0_t6_l0) + m_Vxz_o0_t7_l0;
								m_Vyz_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t5_m0_l0;
								m_Vyz_o0_t1_l0 = (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vyz_o0_t2_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vyz_o0_t3_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vty_o0_t6_m0_l0;
								m_Vyz_o0_t4_l0 = (-vector(Phid_z, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t1_m0_l0;
								m_Vyz_o0_t5_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_Vyz_o0_t6_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtz_o0_t6_m0_l0;
								m_Vyz_o0_t7_l0 = vector(Phid_y, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
								Vyz = ((((((m_Vyz_o0_t0_l0 + m_Vyz_o0_t1_l0) + m_Vyz_o0_t2_l0) + m_Vyz_o0_t3_l0) + m_Vyz_o0_t4_l0) + m_Vyz_o0_t5_l0) + m_Vyz_o0_t6_l0) + m_Vyz_o0_t7_l0;
								m_Vxy_o0_t0_l0 = vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k) * d_Vtx_o0_t4_m0_l0;
								m_Vxy_o0_t1_l0 = (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k)) * d_Vtx_o0_t5_m0_l0;
								m_Vxy_o0_t2_l0 = (-vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t5_m0_l0;
								m_Vxy_o0_t3_l0 = vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k) * d_Vtx_o0_t6_m0_l0;
								m_Vxy_o0_t4_l0 = (-vector(Phid_y, i, j, k) * vector(Psisf, i, j, k)) * d_Vtx_o0_t2_m0_l0;
								m_Vxy_o0_t5_l0 = (-vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_Vxy_o0_t6_l0 = (-vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vty_o0_t6_m0_l0;
								m_Vxy_o0_t7_l0 = vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t1_m0_l0;
								Vxy = ((((((m_Vxy_o0_t0_l0 + m_Vxy_o0_t1_l0) + m_Vxy_o0_t2_l0) + m_Vxy_o0_t3_l0) + m_Vxy_o0_t4_l0) + m_Vxy_o0_t5_l0) + m_Vxy_o0_t6_l0) + m_Vxy_o0_t7_l0;
								Phiu_x = vector(Phid_x, i, j, k);
								Phiu_z = vector(Phid_z, i, j, k);
								Phiu_y = vector(Phid_y, i, j, k);
								Xphi = (-vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) + Phiu_x * vector(Phid_x, i, j, k) + Phiu_y * vector(Phid_y, i, j, k) + Phiu_z * vector(Phid_z, i, j, k);
								KXXf = 0.5 * betax + gammax * Xphi;
								KXf = (-0.5 * sigmax) + 0.5 * betax * Xphi + 0.5 * gammax * (Xphi * Xphi);
								r1_z = zcoord(k);
								r2_z = zcoord(k);
								rorbit_dynamic = MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, ((current_time - tini) - tslow) - torbit)))) * rorbit + MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, ((current_time - tini) - tslow) - torbit)))) * MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, (current_time - tini) - torbit)))) * (rstart + 6.0 / (tslow * tslow * tslow) * (rorbit - rstart) * ((-(((current_time - tini) - torbit) * ((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 3.0) + tslow * (((current_time - tini) - torbit) * ((current_time - tini) - torbit)) / 2.0)) + rstart * MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, (current_time - tini) - torbit))));
								omegaorbit = p_Omega * sqrt(2.0 * mass / ((2.0 * rorbit_dynamic) * (2.0 * rorbit_dynamic) * (2.0 * rorbit_dynamic))) * (MAX(0.0, MIN(1.0, 1.0 - MAX(0.0, MIN(1.0, current_time - torbit)))) * MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, current_time)))) * 4.0 * (2.0 * torbit - current_time) * current_time / ((2.0 * torbit) * (2.0 * torbit)) + MAX(0.0, MIN(1.0, MAX(0.0, MIN(1.0, current_time - torbit)))));
								r2_y = ycoord(j) - 2.0 / (1.0 + mu) * rorbit_dynamic * sin(omegaorbit * current_time);
								r2_x = xcoord(i) - 2.0 / (1.0 + mu) * rorbit_dynamic * cos(omegaorbit * current_time);
								sq_r2 = r2_x * r2_x + r2_y * r2_y + r2_z * r2_z;
								r1_y = ycoord(j) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic * sin(omegaorbit * current_time);
								r1_x = xcoord(i) + 2.0 * mu / (1.0 + mu) * rorbit_dynamic * cos(omegaorbit * current_time);
								sq_r1 = r1_x * r1_x + r1_y * r1_y + r1_z * r1_z;
								Tbar = -(sq_r1 * exp(-((sqrt(sq_r1) - rdonut) * (sqrt(sq_r1) - rdonut)) / (p_sigma * p_sigma)) + mu * (Nstar - 1.0) * sq_r2 * exp(-((sqrt(sq_r2) - rdonut) * (sqrt(sq_r2) - rdonut)) / (p_sigma * p_sigma))) * ((mass * 2.0 / (1.0 + mu)) / massfactor) / (pow((6.28318530717959 * p_sigma), (3.0 / 2.0)));
								d_phi22_o0_t0_m0_l0 = -1.0 / (4.0 * Mpl_cte) * Tbar / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)));
								d_vxplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vxminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vyplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vyminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vzplus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) * KXXf) + sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vzminus_o0_t0_m0_l0 = 1.0 / (KXf - 2.0 * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) * KXXf) * ((-2.0 * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) * KXXf) - sqrt(KXf * (KXf + 2.0 * Xphi * KXXf)));
								d_vorticity_norm_o0_t0_m0_l0 = ((-2.0) * (Vtx * Vtx) - 2.0 * (Vty * Vty)) - 2.0 * (Vtz * Vtz) + 2.0 * (Vxy * Vxy) + 2.0 * (Vxz * Vxz) + 2.0 * (Vyz * Vyz);
								d_d2tphi_o0_t12_m0_l0 = (1.0 / Mpl_cte) / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * Tbar / 4.0;
								d_Chi_t_o0_t0_m0_l0 = KXf * vector(Psisf, i, j, k);
								d_Chi_x_o0_t0_m0_l0 = KXf * vector(Phid_x, i, j, k);
								d_Chi_y_o0_t0_m0_l0 = KXf * vector(Phid_y, i, j, k);
								d_Chi_z_o0_t0_m0_l0 = KXf * vector(Phid_z, i, j, k);
								d_Chi_norm_o0_t0_m0_l0 = (-vector(Chi_t, i, j, k) * vector(Chi_t, i, j, k)) + vector(Chi_x, i, j, k) * vector(Chi_x, i, j, k) + vector(Chi_y, i, j, k) * vector(Chi_y, i, j, k) + vector(Chi_z, i, j, k) * vector(Chi_z, i, j, k);
								m_phi22_o0_t1_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t6_m0_l0;
								m_phi22_o0_t2_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vty_o0_t6_m0_l0;
								m_phi22_o0_t3_l0 = KXf / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtz_o0_t6_m0_l0;
								m_phi22_o0_t4_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_x, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t2_m0_l0;
								m_phi22_o0_t5_l0 = (2.0 * KXXf * (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t6_m0_l0;
								m_phi22_o0_t6_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_y, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t1_m0_l0;
								m_phi22_o0_t7_l0 = (4.0 * KXXf * vector(Phid_x, i, j, k) * vector(Phid_y, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t7_m0_l0;
								m_phi22_o0_t8_l0 = (2.0 * KXXf * (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vty_o0_t6_m0_l0;
								m_phi22_o0_t9_l0 = (-4.0) * KXXf * vector(Psisf, i, j, k) * vector(Phid_z, i, j, k) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtx_o0_t0_m0_l0;
								m_phi22_o0_t10_l0 = (4.0 * KXXf * vector(Phid_x, i, j, k) * vector(Phid_z, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t10_m0_l0;
								m_phi22_o0_t11_l0 = (4.0 * KXXf * vector(Phid_y, i, j, k) * vector(Phid_z, i, j, k)) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_phi22_o0_t11_m0_l0;
								m_phi22_o0_t12_l0 = (2.0 * KXXf * (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k))) / (KXf - 2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k))) * d_Vtz_o0_t6_m0_l0;
								m_d2tphi_o0_t0_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_x, i, j, k) * vector(Phid_x, i, j, k))) * d_Vtx_o0_t6_m0_l0;
								m_d2tphi_o0_t1_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_y, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t5_m0_l0;
								m_d2tphi_o0_t2_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Phid_x, i, j, k)) * d_Vtx_o0_t4_m0_l0;
								m_d2tphi_o0_t3_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_x, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t2_m0_l0;
								m_d2tphi_o0_t4_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_y, i, j, k) * vector(Phid_y, i, j, k))) * d_Vty_o0_t6_m0_l0;
								m_d2tphi_o0_t5_l0 = (-4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Phid_y, i, j, k)) * d_Vty_o0_t5_m0_l0;
								m_d2tphi_o0_t6_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_y, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t1_m0_l0;
								m_d2tphi_o0_t7_l0 = (-2.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * (vector(Phid_z, i, j, k) * vector(Phid_z, i, j, k))) * d_Vtz_o0_t6_m0_l0;
								m_d2tphi_o0_t8_l0 = 4.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXXf * vector(Phid_z, i, j, k) * vector(Psisf, i, j, k) * d_Vtx_o0_t0_m0_l0;
								m_d2tphi_o0_t9_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vtz_o0_t6_m0_l0;
								m_d2tphi_o0_t10_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vty_o0_t6_m0_l0;
								m_d2tphi_o0_t11_l0 = (-1.0 / (2.0 * KXXf * (vector(Psisf, i, j, k) * vector(Psisf, i, j, k)) - KXf) * KXf) * d_Vtx_o0_t6_m0_l0;
								vector(phi22, i, j, k) = (((((((((((d_phi22_o0_t0_m0_l0 + m_phi22_o0_t1_l0) + m_phi22_o0_t2_l0) + m_phi22_o0_t3_l0) + m_phi22_o0_t4_l0) + m_phi22_o0_t5_l0) + m_phi22_o0_t6_l0) + m_phi22_o0_t7_l0) + m_phi22_o0_t8_l0) + m_phi22_o0_t9_l0) + m_phi22_o0_t10_l0) + m_phi22_o0_t11_l0) + m_phi22_o0_t12_l0;
								vector(vxplus, i, j, k) = d_vxplus_o0_t0_m0_l0;
								vector(vxminus, i, j, k) = d_vxminus_o0_t0_m0_l0;
								vector(vyplus, i, j, k) = d_vyplus_o0_t0_m0_l0;
								vector(vyminus, i, j, k) = d_vyminus_o0_t0_m0_l0;
								vector(vzplus, i, j, k) = d_vzplus_o0_t0_m0_l0;
								vector(vzminus, i, j, k) = d_vzminus_o0_t0_m0_l0;
								vector(vorticity_norm, i, j, k) = d_vorticity_norm_o0_t0_m0_l0;
								vector(d2tphi, i, j, k) = (((((((((((m_d2tphi_o0_t0_l0 + m_d2tphi_o0_t1_l0) + m_d2tphi_o0_t2_l0) + m_d2tphi_o0_t3_l0) + m_d2tphi_o0_t4_l0) + m_d2tphi_o0_t5_l0) + m_d2tphi_o0_t6_l0) + m_d2tphi_o0_t7_l0) + m_d2tphi_o0_t8_l0) + m_d2tphi_o0_t9_l0) + m_d2tphi_o0_t10_l0) + m_d2tphi_o0_t11_l0) + d_d2tphi_o0_t12_m0_l0;
								vector(Chi_t, i, j, k) = d_Chi_t_o0_t0_m0_l0;
								vector(Chi_x, i, j, k) = d_Chi_x_o0_t0_m0_l0;
								vector(Chi_y, i, j, k) = d_Chi_y_o0_t0_m0_l0;
								vector(Chi_z, i, j, k) = d_Chi_z_o0_t0_m0_l0;
								vector(Chi_norm, i, j, k) = d_Chi_norm_o0_t0_m0_l0;
							}
						}
					}
				}
			}
		}
		

	}


	//Output
	t_output->start();
	if (ln == hierarchy->getFinestLevelNumber() && viz_mesh_dump_interval > 0) {
		if (previous_iteration < next_mesh_dump_iteration && outputCycle >= next_mesh_dump_iteration) {
			d_visit_data_writer->writePlotData(hierarchy, outputCycle, new_time);
			next_mesh_dump_iteration = next_mesh_dump_iteration + viz_mesh_dump_interval;
			while (outputCycle >= next_mesh_dump_iteration) {
				next_mesh_dump_iteration = next_mesh_dump_iteration + viz_mesh_dump_interval;
			}
		}
	}
	//Slicer output
	if (ln == hierarchy->getFinestLevelNumber() && d_slicer_output_period.size() > 0) {
		int i = 0;
		for (std::vector<std::shared_ptr<SlicerDataWriter> >::iterator it = d_sliceWriters.begin(); it != d_sliceWriters.end(); ++it) {
			if (d_slicer_output_period[i] > 0 && previous_iteration < next_slice_dump_iteration[i] && outputCycle >= next_slice_dump_iteration[i]) {
				(*it)->writePlotData(hierarchy, outputCycle, new_time);
				next_slice_dump_iteration[i] = next_slice_dump_iteration[i] + d_slicer_output_period[i];
				while (outputCycle >= next_slice_dump_iteration[i]) {
					next_slice_dump_iteration[i] = next_slice_dump_iteration[i] + d_slicer_output_period[i];
				}
			}
			i++;
		}
	}

	//Spherical output
	if (ln == hierarchy->getFinestLevelNumber() && d_sphere_output_period.size() > 0) {
		int i = 0;
		for (std::vector<std::shared_ptr<SphereDataWriter> >::iterator it = d_sphereWriters.begin(); it != d_sphereWriters.end(); ++it) {
			if (d_sphere_output_period[i] > 0 && previous_iteration < next_sphere_dump_iteration[i] && outputCycle >= next_sphere_dump_iteration[i]) {
				(*it)->writePlotData(hierarchy, outputCycle, new_time);
				next_sphere_dump_iteration[i] = next_sphere_dump_iteration[i] + d_sphere_output_period[i];
				while (outputCycle >= next_sphere_dump_iteration[i]) {
					next_sphere_dump_iteration[i] = next_sphere_dump_iteration[i] + d_sphere_output_period[i];
				}
			}
			i++;
		}
	}

	//Integration output
	if (ln == hierarchy->getFinestLevelNumber() && d_integration_output_period.size() > 0) {
		int i = 0;
		for (std::vector<std::shared_ptr<IntegrateDataWriter> >::iterator it = d_integrateDataWriters.begin(); it != d_integrateDataWriters.end(); ++it) {
			if (d_integration_output_period[i] > 0 && previous_iteration < next_integration_dump_iteration[i] && outputCycle >= next_integration_dump_iteration[i]) {
				(*it)->writePlotData(hierarchy, outputCycle, new_time);
				next_integration_dump_iteration[i] = next_integration_dump_iteration[i] + d_integration_output_period[i];
				while (outputCycle >= next_integration_dump_iteration[i]) {
					next_integration_dump_iteration[i] = next_integration_dump_iteration[i] + d_integration_output_period[i];
				}
			}
			i++;
		}
	}

	//Point output
	if (ln == hierarchy->getFinestLevelNumber() && d_point_output_period.size() > 0) {
		int i = 0;
		for (std::vector<std::shared_ptr<PointDataWriter> >::iterator it = d_pointDataWriters.begin(); it != d_pointDataWriters.end(); ++it) {
			if (d_point_output_period[i] > 0 && previous_iteration < next_point_dump_iteration[i] && outputCycle >= next_point_dump_iteration[i]) {
				(*it)->writePlotData(hierarchy, outputCycle, new_time);
				next_point_dump_iteration[i] = next_point_dump_iteration[i] + d_point_output_period[i];
				while (outputCycle >= next_point_dump_iteration[i]) {
					next_point_dump_iteration[i] = next_point_dump_iteration[i] + d_point_output_period[i];
				}
			}
			i++;
		}
	}


	t_output->stop();

	if (mpi.getRank() == 0 && d_output_interval > 0 ) {
		if (previous_iteration < next_console_output && outputCycle >= next_console_output) {
			int currentLevelNumber = ln;
			while (currentLevelNumber > 0) {
				currentLevelNumber--;
				cout <<"  ";
			}

			cout << "Level "<<ln<<". Iteration " << current_iteration[ln]<<". Time "<<current_time<<"."<< endl;
			if (ln == hierarchy->getFinestLevelNumber()) {
				next_console_output = next_console_output + d_output_interval;
				while (outputCycle >= next_console_output) {
					next_console_output = next_console_output + d_output_interval;
				}
			
			}
		}
	}

	if (d_timer_output_interval > 0 ) {
		if (previous_iteration < next_timer_output && outputCycle >= next_timer_output) {
			if (ln == hierarchy->getFinestLevelNumber()) {
				//Print timers
				if (mpi.getRank() == 0) {
					tbox::TimerManager::getManager()->print(cout);
				}
				else {
					if (ln == hierarchy->getFinestLevelNumber()) {
						//Dispose other processor timers
						//SAMRAI needs all processors run tbox::TimerManager::getManager()->print, otherwise it hungs
						std::ofstream ofs;
						ofs.setstate(std::ios_base::badbit);
						tbox::TimerManager::getManager()->print(ofs);
					}
				}
				next_timer_output = next_timer_output + d_timer_output_interval;
				while (outputCycle >= next_timer_output) {
					next_timer_output = next_timer_output + d_timer_output_interval;
				}
			
			}
		}
	}

	return simPlat_dt;
}

/*
 * Checks the finalization conditions              
 */
bool Problem::checkFinalization(const double current_time, const double simPlat_dt)
{
	if (greaterEq(current_time, tend)) { 
		return true;
	}
	return false;
	
	

}

void Problem::putToRestart(MainRestartData& mrd) {
	mrd.setNextSliceDumpIteration(next_slice_dump_iteration);
	mrd.setNextSphereDumpIteration(next_sphere_dump_iteration);
	mrd.setNextMeshDumpIteration(next_mesh_dump_iteration);
	mrd.setNextIntegrationDumpIteration(next_integration_dump_iteration);
	mrd.setNextPointDumpIteration(next_point_dump_iteration);

	mrd.setCurrentIteration(current_iteration);
	mrd.setNextConsoleOutputIteration(next_console_output);
	mrd.setNextTimerOutputIteration(next_timer_output);
}

void Problem::getFromRestart(MainRestartData& mrd) {
	next_slice_dump_iteration = mrd.getNextSliceDumpIteration();
	next_sphere_dump_iteration = mrd.getNextSphereDumpIteration();
	next_mesh_dump_iteration = mrd.getNextMeshDumpIteration();
	next_integration_dump_iteration = mrd.getNextIntegrationDumpIteration();
	next_point_dump_iteration = mrd.getNextPointDumpIteration();

	current_iteration = mrd.getCurrentIteration();
	next_console_output = mrd.getNextConsoleOutputIteration();
	next_timer_output = mrd.getNextTimerOutputIteration();
}

void Problem::allocateAfterRestart() {
	for (int il = 0; il < d_patch_hierarchy->getNumberOfLevels(); il++) {
		std::shared_ptr< hier::PatchLevel > level(d_patch_hierarchy->getPatchLevel(il));
		level->allocatePatchData(d_mask_id);
		level->allocatePatchData(d_interior_regridding_value_id);
		level->allocatePatchData(d_nonSync_regridding_tag_id);
		level->allocatePatchData(d_interior_i_id);
		level->allocatePatchData(d_interior_j_id);
		level->allocatePatchData(d_interior_k_id);
		level->allocatePatchData(d_phi22_id);
		level->allocatePatchData(d_vxplus_id);
		level->allocatePatchData(d_vxminus_id);
		level->allocatePatchData(d_vyplus_id);
		level->allocatePatchData(d_vyminus_id);
		level->allocatePatchData(d_vzplus_id);
		level->allocatePatchData(d_vzminus_id);
		level->allocatePatchData(d_vorticity_norm_id);
		level->allocatePatchData(d_d2tphi_id);
		level->allocatePatchData(d_Chi_t_id);
		level->allocatePatchData(d_Chi_x_id);
		level->allocatePatchData(d_Chi_y_id);
		level->allocatePatchData(d_Chi_z_id);
		level->allocatePatchData(d_Chi_norm_id);
		level->allocatePatchData(d_rk1Psisf_id);
		level->allocatePatchData(d_rk1phi_id);
		level->allocatePatchData(d_rk1Phid_x_id);
		level->allocatePatchData(d_rk1Phid_y_id);
		level->allocatePatchData(d_rk1Phid_z_id);
		level->allocatePatchData(d_rk1Psi_id);
		level->allocatePatchData(d_rk2Psisf_id);
		level->allocatePatchData(d_rk2phi_id);
		level->allocatePatchData(d_rk2Phid_x_id);
		level->allocatePatchData(d_rk2Phid_y_id);
		level->allocatePatchData(d_rk2Phid_z_id);
		level->allocatePatchData(d_rk2Psi_id);
		level->allocatePatchData(d_rk3Psisf_id);
		level->allocatePatchData(d_rk3phi_id);
		level->allocatePatchData(d_rk3Phid_x_id);
		level->allocatePatchData(d_rk3Phid_y_id);
		level->allocatePatchData(d_rk3Phid_z_id);
		level->allocatePatchData(d_rk3Psi_id);
		level->allocatePatchData(d_Psisf_p_id);
		level->allocatePatchData(d_phi_p_id);
		level->allocatePatchData(d_Phid_x_p_id);
		level->allocatePatchData(d_Phid_y_p_id);
		level->allocatePatchData(d_Phid_z_p_id);
		level->allocatePatchData(d_Psi_p_id);
	}

}

/*
 *  Cell tagging routine - tag cells that require refinement based on a provided condition. 
 */
void Problem::applyGradientDetector(
   const std::shared_ptr< hier::PatchHierarchy >& hierarchy, 
   const int level_number,
   const double time, 
   const int tag_index,
   const bool initial_time,
   const bool uses_richardson_extrapolation_too) 
{
	if (d_regridding && level_number + 1 >= d_regridding_min_level && level_number + 1 <= d_regridding_max_level) {
		hier::VariableDatabase *vdb = hier::VariableDatabase::getDatabase();
		if (!(vdb->checkVariableExists(d_regridding_field))) {
			TBOX_ERROR(d_object_name << ": Regridding field selected not found:n"					<<  d_regridding_field<<  "n");
		}

		std::shared_ptr< hier::PatchLevel > level(hierarchy->getPatchLevel(level_number));

		for (hier::PatchLevel::iterator ip(level->begin()); ip != level->end(); ++ip) {
			const std::shared_ptr< hier::Patch >& patch = *ip;
			int* tags = ((pdat::CellData<int> *) patch->getPatchData(tag_index).get())->getPointer();
			int* regridding_tag = ((pdat::NodeData<int> *) patch->getPatchData(d_nonSync_regridding_tag_id).get())->getPointer();
			const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
			const double* dx  = patch_geom->getDx();
			const hier::Index tfirst = patch->getPatchData(tag_index)->getGhostBox().lower();
			const hier::Index tlast  = patch->getPatchData(tag_index)->getGhostBox().upper();
			const hier::Index boxfirst = patch->getBox().lower();
			const hier::Index boxlast  = patch->getBox().upper();
			int ilast = boxlast(0)-boxfirst(0)+2+2*d_ghost_width;
			int itlast = tlast(0)-tfirst(0)+1;
			int jlast = boxlast(1)-boxfirst(1)+2+2*d_ghost_width;
			int jtlast = tlast(1)-tfirst(1)+1;
			int klast = boxlast(2)-boxfirst(2)+2+2*d_ghost_width;
			int ktlast = tlast(2)-tfirst(2)+1;
			for(int index2 = 0; index2 < (tlast(2)-tfirst(2))+1; index2++) {
				for(int index1 = 0; index1 < (tlast(1)-tfirst(1))+1; index1++) {
					for(int index0 = 0; index0 < (tlast(0)-tfirst(0))+1; index0++) {
						vectorT(tags,index0, index1, index2) = 0;
					}
				}
			}
			for(int index2 = 0; index2 < klast; index2++) {
				for(int index1 = 0; index1 < jlast; index1++) {
					for(int index0 = 0; index0 < ilast; index0++) {
						vector(regridding_tag,index0, index1, index2) = 0;
					}
				}
			}
			if (vdb->checkVariableExists(d_regridding_field)) {
				//Mesh
				if (d_regridding_type == "GRADIENT") {
					int regrid_field_id = vdb->getVariable(d_regridding_field)->getInstanceIdentifier();
					double* regrid_field = ((pdat::NodeData<double> *) patch->getPatchData(regrid_field_id).get())->getPointer();
					double* regridding_value = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
					for(int index2 = 0; index2 < klast - 1; index2++) {
						for(int index1 = 0; index1 < jlast - 1; index1++) {
							for(int index0 = 0; index0 < ilast - 1; index0++) {
								if (vector(regrid_field, index0, index1, index2)!=0) {
									if ((fabs(vector(regrid_field, index0+1+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)) > d_regridding_compressionFactor * MIN(fabs(vector(regrid_field, index0+2+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+1+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)) + d_regridding_mOffset * pow(dx[0], 2), fabs(vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0-1+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)) + d_regridding_mOffset * pow(dx[0], 2)) ) ||(fabs(vector(regrid_field, index0+d_ghost_width, index1+1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)) > d_regridding_compressionFactor * MIN(fabs(vector(regrid_field, index0+d_ghost_width, index1+2+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+1+d_ghost_width, index2+d_ghost_width)) + d_regridding_mOffset * pow(dx[1], 2), fabs(vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1-1+d_ghost_width, index2+d_ghost_width)) + d_regridding_mOffset * pow(dx[1], 2)) ) ||(fabs(vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+1+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)) > d_regridding_compressionFactor * MIN(fabs(vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+1+d_ghost_width)) + d_regridding_mOffset * pow(dx[2], 2), fabs(vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2+d_ghost_width)-vector(regrid_field, index0+d_ghost_width, index1+d_ghost_width, index2-1+d_ghost_width)) + d_regridding_mOffset * pow(dx[2], 2)) ) ) {
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags,index0 - d_ghost_width, index1 - d_ghost_width, index2 - d_ghost_width) = 1;
										}
										vector(regridding_tag,index0, index1, index2) = 1;
										//SAMRAI tagging
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
										}
										//Informative tagging
										if (index0 > 0 && index1 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0-1,index1-1,index2-1) != 1)
												vector(regridding_tag,index0-1,index1-1,index2-1) = 1;
										}
										if (index1 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0,index1-1,index2-1) != 1)
												vector(regridding_tag,index0,index1-1,index2-1) = 1;
										}
										if (index0 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0-1,index1,index2-1) != 1)
												vector(regridding_tag,index0-1,index1,index2-1) = 1;
										}
										if (index2 > 0) {
											if (vector(regridding_tag,index0,index1,index2-1) != 1)
												vector(regridding_tag,index0,index1,index2-1) = 1;
										}
										if (index0 > 0 && index1 > 0) {
											if (vector(regridding_tag,index0-1,index1-1,index2) != 1)
												vector(regridding_tag,index0-1,index1-1,index2) = 1;
										}
										if (index0 > 0) {
											if (vector(regridding_tag,index0-1,index1,index2) != 1)
												vector(regridding_tag,index0-1,index1,index2) = 1;
										}
										if (index1 > 0) {
											if (vector(regridding_tag,index0,index1-1,index2) != 1)
												vector(regridding_tag,index0,index1-1,index2) = 1;
										}
										if (d_regridding_buffer > 0) {
											int distance;
											for(int index2b = MAX(0, index2 - d_regridding_buffer - 1); index2b < MIN(index2 + d_regridding_buffer + 1, klast); index2b++) {
												for(int index1b = MAX(0, index1 - d_regridding_buffer - 1); index1b < MIN(index1 + d_regridding_buffer + 1, jlast); index1b++) {
													for(int index0b = MAX(0, index0 - d_regridding_buffer - 1); index0b < MIN(index0 + d_regridding_buffer + 1, ilast); index0b++) {
														int distx = (index0b - index0);
														if (distx < 0) {
															distx++;
														}
														int disty = (index1b - index1);
														if (disty < 0) {
															disty++;
														}
														int distz = (index2b - index2);
														if (distz < 0) {
															distz++;
														}
														distance = 1 + MAX(MAX(abs(distx), abs(disty)), abs(distz));
														if (index0b >= d_ghost_width && index0b < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1b >= d_ghost_width && index1b < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2b >= d_ghost_width && index2b < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
															vectorT(tags,index0b-d_ghost_width,index1b-d_ghost_width,index2b-d_ghost_width) = 1;
														}
														if (vector(regridding_tag,index0b,index1b,index2b) == 0 || vector(regridding_tag,index0b,index1b,index2b) > distance) {
															vector(regridding_tag,index0b,index1b,index2b) = distance;
														}
													}
												}
											}
										}
			
									}
								}
							}
						}
					}
		
				} else {
					if (d_regridding_type == "FUNCTION") {
						int regrid_field_id = vdb->getVariable(d_regridding_field)->getInstanceIdentifier();
						double* regrid_field = ((pdat::NodeData<double> *) patch->getPatchData(regrid_field_id).get())->getPointer();
						double* regridding_value = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
						for(int index2 = 0; index2 < klast; index2++) {
							for(int index1 = 0; index1 < jlast; index1++) {
								for(int index0 = 0; index0 < ilast; index0++) {
									vector(regridding_value, index0, index1, index2) = vector(regrid_field, index0, index1, index2);
									if (vector(regrid_field, index0, index1, index2) > d_regridding_threshold) {
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags,index0 - d_ghost_width, index1 - d_ghost_width, index2 - d_ghost_width) = 1;
										}
										vector(regridding_tag,index0, index1, index2) = 1;
										//SAMRAI tagging
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
										}
										if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width) = 1;
										}
										if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
											vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
										}
										//Informative tagging
										if (index0 > 0 && index1 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0-1,index1-1,index2-1) != 1)
												vector(regridding_tag,index0-1,index1-1,index2-1) = 1;
										}
										if (index1 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0,index1-1,index2-1) != 1)
												vector(regridding_tag,index0,index1-1,index2-1) = 1;
										}
										if (index0 > 0 && index2 > 0) {
											if (vector(regridding_tag,index0-1,index1,index2-1) != 1)
												vector(regridding_tag,index0-1,index1,index2-1) = 1;
										}
										if (index2 > 0) {
											if (vector(regridding_tag,index0,index1,index2-1) != 1)
												vector(regridding_tag,index0,index1,index2-1) = 1;
										}
										if (index0 > 0 && index1 > 0) {
											if (vector(regridding_tag,index0-1,index1-1,index2) != 1)
												vector(regridding_tag,index0-1,index1-1,index2) = 1;
										}
										if (index0 > 0) {
											if (vector(regridding_tag,index0-1,index1,index2) != 1)
												vector(regridding_tag,index0-1,index1,index2) = 1;
										}
										if (index1 > 0) {
											if (vector(regridding_tag,index0,index1-1,index2) != 1)
												vector(regridding_tag,index0,index1-1,index2) = 1;
										}
										if (d_regridding_buffer > 0) {
											int distance;
											for(int index2b = MAX(0, index2 - d_regridding_buffer - 1); index2b < MIN(index2 + d_regridding_buffer + 1, klast); index2b++) {
												for(int index1b = MAX(0, index1 - d_regridding_buffer - 1); index1b < MIN(index1 + d_regridding_buffer + 1, jlast); index1b++) {
													for(int index0b = MAX(0, index0 - d_regridding_buffer - 1); index0b < MIN(index0 + d_regridding_buffer + 1, ilast); index0b++) {
														int distx = (index0b - index0);
														if (distx < 0) {
															distx++;
														}
														int disty = (index1b - index1);
														if (disty < 0) {
															disty++;
														}
														int distz = (index2b - index2);
														if (distz < 0) {
															distz++;
														}
														distance = 1 + MAX(MAX(abs(distx), abs(disty)), abs(distz));
														if (index0b >= d_ghost_width && index0b < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1b >= d_ghost_width && index1b < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2b >= d_ghost_width && index2b < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
															vectorT(tags,index0b-d_ghost_width,index1b-d_ghost_width,index2b-d_ghost_width) = 1;
														}
														if (vector(regridding_tag,index0b,index1b,index2b) == 0 || vector(regridding_tag,index0b,index1b,index2b) > distance) {
															vector(regridding_tag,index0b,index1b,index2b) = distance;
														}
													}
												}
											}
										}
									}
								}
							}
						}
			
					} else {
						if (d_regridding_type == "SHADOW") {
							if (!initial_time) {
								if (!(vdb->checkVariableExists(d_regridding_field_shadow))) {
									TBOX_ERROR(d_object_name << ": Regridding field selected not found:" <<  d_regridding_field_shadow<<  "");
								}
								int regrid_field_id = vdb->getVariable(d_regridding_field)->getInstanceIdentifier();
								double* regrid_field1 = ((pdat::NodeData<double> *) patch->getPatchData(regrid_field_id).get())->getPointer();
								int regrid_field_shadow_id = vdb->getVariable(d_regridding_field_shadow)->getInstanceIdentifier();
								double* regrid_field2 = ((pdat::NodeData<double> *) patch->getPatchData(regrid_field_shadow_id).get())->getPointer();
								double* regridding_value = ((pdat::NodeData<double> *) patch->getPatchData(d_interior_regridding_value_id).get())->getPointer();
								for(int index2 = 0; index2 < klast; index2++) {
									for(int index1 = 0; index1 < jlast; index1++) {
										for(int index0 = 0; index0 < ilast; index0++) {
					
											double error = 2 * fabs(vector(regrid_field1, index0, index1, index2) - vector(regrid_field2, index0, index1, index2))/fabs(vector(regrid_field1, index0, index1, index2) + vector(regrid_field2, index0, index1, index2));
											vector(regridding_value, index0, index1, index2) = error;
											if (error > d_regridding_error) {
												if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags,index0 - d_ghost_width, index1 - d_ghost_width, index2 - d_ghost_width) = 1;
												}
												vector(regridding_tag,index0, index1, index2) = 1;
												//SAMRAI tagging
												if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
												}
												if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width - 1) = 1;
												}
												if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
												}
												if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 > d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width, index1-d_ghost_width, index2-d_ghost_width - 1) = 1;
												}
												if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
												}
												if (index0 > d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 >= d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width - 1, index1-d_ghost_width, index2-d_ghost_width) = 1;
												}
												if (index0 >= d_ghost_width && index0 < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1 > d_ghost_width && index1 < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2 >= d_ghost_width && index2 < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
													vectorT(tags, index0-d_ghost_width, index1-d_ghost_width - 1, index2-d_ghost_width) = 1;
												}
												//Informative tagging
												if (index0 > 0 && index1 > 0 && index2 > 0) {
													if (vector(regridding_tag,index0-1,index1-1,index2-1) != 1)
														vector(regridding_tag,index0-1,index1-1,index2-1) = 1;
												}
												if (index1 > 0 && index2 > 0) {
													if (vector(regridding_tag,index0,index1-1,index2-1) != 1)
														vector(regridding_tag,index0,index1-1,index2-1) = 1;
												}
												if (index0 > 0 && index2 > 0) {
													if (vector(regridding_tag,index0-1,index1,index2-1) != 1)
														vector(regridding_tag,index0-1,index1,index2-1) = 1;
												}
												if (index2 > 0) {
													if (vector(regridding_tag,index0,index1,index2-1) != 1)
														vector(regridding_tag,index0,index1,index2-1) = 1;
												}
												if (index0 > 0 && index1 > 0) {
													if (vector(regridding_tag,index0-1,index1-1,index2) != 1)
														vector(regridding_tag,index0-1,index1-1,index2) = 1;
												}
												if (index0 > 0) {
													if (vector(regridding_tag,index0-1,index1,index2) != 1)
														vector(regridding_tag,index0-1,index1,index2) = 1;
												}
												if (index1 > 0) {
													if (vector(regridding_tag,index0,index1-1,index2) != 1)
														vector(regridding_tag,index0,index1-1,index2) = 1;
												}
												if (d_regridding_buffer > 0) {
													int distance;
													for(int index2b = MAX(0, index2 - d_regridding_buffer - 1); index2b < MIN(index2 + d_regridding_buffer + 1, klast); index2b++) {
														for(int index1b = MAX(0, index1 - d_regridding_buffer - 1); index1b < MIN(index1 + d_regridding_buffer + 1, jlast); index1b++) {
															for(int index0b = MAX(0, index0 - d_regridding_buffer - 1); index0b < MIN(index0 + d_regridding_buffer + 1, ilast); index0b++) {
																int distx = (index0b - index0);
																if (distx < 0) {
																	distx++;
																}
																int disty = (index1b - index1);
																if (disty < 0) {
																	disty++;
																}
																int distz = (index2b - index2);
																if (distz < 0) {
																	distz++;
																}
																distance = 1 + MAX(MAX(abs(distx), abs(disty)), abs(distz));
																if (index0b >= d_ghost_width && index0b < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1b >= d_ghost_width && index1b < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2b >= d_ghost_width && index2b < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
																	vectorT(tags,index0b-d_ghost_width,index1b-d_ghost_width,index2b-d_ghost_width) = 1;
																}
																if (vector(regridding_tag,index0b,index1b,index2b) == 0 || vector(regridding_tag,index0b,index1b,index2b) > distance) {
																	vector(regridding_tag,index0b,index1b,index2b) = distance;
																}
															}
														}
													}
												}
					
											}
										}
									}
								}
					
							}
				
						}
					}
				}
			}
		}
		//Buffer synchronization if needed
		if (d_regridding_buffer > d_ghost_width) {
			d_tagging_fill->createSchedule(level)->fillData(0, false);
		}
		if (d_regridding_buffer > 0) {
			for (hier::PatchLevel::iterator ip(level->begin()); ip != level->end(); ++ip) {
				const std::shared_ptr< hier::Patch >& patch = *ip;
				int* tags = ((pdat::CellData<int> *) patch->getPatchData(tag_index).get())->getPointer();
				int* regridding_tag = ((pdat::NodeData<int> *) patch->getPatchData(d_nonSync_regridding_tag_id).get())->getPointer();
				const std::shared_ptr<geom::CartesianPatchGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianPatchGeometry, hier::PatchGeometry>(patch->getPatchGeometry()));
				const double* dx  = patch_geom->getDx();
				const hier::Index tfirst = patch->getPatchData(tag_index)->getGhostBox().lower();
				const hier::Index tlast  = patch->getPatchData(tag_index)->getGhostBox().upper();
				const hier::Index boxfirst = patch->getBox().lower();
				const hier::Index boxlast  = patch->getBox().upper();
				int ilast = boxlast(0)-boxfirst(0)+2+2*d_ghost_width;
				int itlast = tlast(0)-tfirst(0)+1;
				int jlast = boxlast(1)-boxfirst(1)+2+2*d_ghost_width;
				int jtlast = tlast(1)-tfirst(1)+1;
				int klast = boxlast(2)-boxfirst(2)+2+2*d_ghost_width;
				int ktlast = tlast(2)-tfirst(2)+1;
	
				for(int index2 = 0; index2 < klast; index2++) {
					for(int index1 = 0; index1 < jlast; index1++) {
						for(int index0 = 0; index0 < ilast; index0++) {
	
							int value = vector(regridding_tag, index0, index1, index2);
							if (value > 0 && value < 1 + d_regridding_buffer) {
								int buffer_left = 1 + d_regridding_buffer - value;
								int distance;
								for(int index2b = MAX(0, index2 - buffer_left); index2b < MIN(index2 + buffer_left + 1, klast); index2b++) {
									for(int index1b = MAX(0, index1 - buffer_left); index1b < MIN(index1 + buffer_left + 1, jlast); index1b++) {
										for(int index0b = MAX(0, index0 - buffer_left); index0b < MIN(index0 + buffer_left + 1, ilast); index0b++) {
								
											distance = MAX(MAX(abs(index0b - index0), abs(index1b - index1)), abs(index2b - index2));
											if (distance > 0 && index0b >= d_ghost_width && index0b < (tlast(0)-tfirst(0))+1 + d_ghost_width && index1b >= d_ghost_width && index1b < (tlast(1)-tfirst(1))+1 + d_ghost_width && index2b >= d_ghost_width && index2b < (tlast(2)-tfirst(2))+1 + d_ghost_width) {
												vectorT(tags, index0b - d_ghost_width, index1b - d_ghost_width, index2b - d_ghost_width) = 1;
											}
											if (distance > 0 && (vector(regridding_tag,index0b, index1b, index2b) == 0 || vector(regridding_tag, index0b, index1b, index2b) > value + distance)) {
												vector(regridding_tag, index0b, index1b, index2b) = value + distance;
											}
										}
									}
								}
								
							}
						}
					}
				}
	
			}
		}
	}
}



