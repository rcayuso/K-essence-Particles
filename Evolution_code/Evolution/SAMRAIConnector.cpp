#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <execinfo.h>
#include "string.h"

#include "Problem.h"

#include "SAMRAI/tbox/SAMRAIManager.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"
#include "SAMRAI/tbox/RestartManager.h"
#include "SAMRAI/tbox/SAMRAI_MPI.h"
#include "SAMRAI/tbox/PIO.h"
#include "SAMRAI/tbox/Utilities.h"
#include "SAMRAI/tbox/Timer.h"
#include "SAMRAI/tbox/TimerManager.h"
#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/tbox/BalancedDepthFirstTree.h"
#include "SAMRAI/mesh/TileClustering.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/mesh/TreeLoadBalancer.h"
#include "SAMRAI/mesh/ChopAndPackLoadBalancer.h"
#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "MainRestartData.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "TimeRefinementIntegrator.h"
#include "SAMRAI/algs/TimeRefinementLevelStrategy.h"
#include "SAMRAI/appu/VisItDataWriter.h"


const int MPI_MASTER = 0;

// SAMRAI namespaces
using namespace std;
using namespace SAMRAI;

//Global variables
Problem* problem;

int n_at_commands;
int iteration_num;
double loop_time;
double dt;
int output_interval;
int timer_output_interval;
int restore_num = 0;
bool is_from_restart = false;
string viz_mesh_dump_dirname;
vector<string> full_mesh_writer_variables;
int viz_mesh_dump_interval;

vector<std::shared_ptr<IntegrateDataWriter > > integrateDataWriters;
vector<int> integralIntervals;
vector<set<string> > integralVariables;
vector<bool> integralAnalysis;
vector<std::shared_ptr<PointDataWriter > > pointDataWriters;
vector<int> pointIntervals;
vector<set<string> > pointVariables;
vector<bool> pointAnalysis;


vector<std::shared_ptr<SlicerDataWriter > > sliceWriters;
vector<int> sliceIntervals;
vector<set<string> > sliceVariables;
vector<std::shared_ptr<SphereDataWriter > > sphereWriters;
vector<int> sphereIntervals;
vector<set<string> > sphereVariables;
vector<bool> sliceAnalysis;
vector<bool> sphereAnalysis;


ostringstream output_information;

int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

struct sigcontext ctx;

void bt_sighandler(int sig) {

  void *trace[16];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  if (sig == SIGSEGV)
    printf("Got signal %d, faulty address is %p, "
           "from %p\n", sig, (void *)ctx.cr2, (void *)ctx.rip);
  else
    printf("Got signal %d\n", sig);

  trace_size = backtrace(trace, 16);
  /* overwrite sigaction with caller's address */
  trace[1] = (void *)ctx.rip;
  messages = backtrace_symbols(trace, trace_size);
  /* skip first stack frame (points here) */
  printf("[bt] Execution path:\n");
  for (i=1; i<trace_size; ++i)
  {
    printf("[bt] #%d %s\n", i, messages[i]);

    /* find first occurence of '(' or ' ' in message[i] and assume
     * everything before that is the file name. (Don't go beyond 0 though
     * (string terminator)*/
    size_t p = 0;
    while(messages[i][p] != '(' && messages[i][p] != ' '
            && messages[i][p] != 0)
        ++p;

    char syscom[256];
    sprintf(syscom,"addr2line %p -e %.*s", trace[i], (int) p, messages[i]);
        //last parameter is the file name of the symbol
    int err = system(syscom);
  }

  exit(0);
}

int main( int argc, char* argv[] )
{
	tbox::SAMRAI_MPI::init(&argc, &argv);
	tbox::SAMRAIManager::initialize();
	tbox::SAMRAIManager::startup();
	const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());
	tbox::SAMRAIManager::setMaxNumberPatchDataEntries(100);
	int my_proc = mpi.getRank();

	if (argc != 2) {
		TBOX_ERROR("Usage: "<<argv[0]<<" <parameter file>");
		return -1;
	}

	//Signal handling
  	struct sigaction sa;
  	sa.sa_handler = (void (*)(int))bt_sighandler;
  	sigemptyset(&sa.sa_mask);
  	sa.sa_flags = SA_RESTART;
  	sigaction(SIGSEGV, &sa, NULL);
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGQUIT, &sa, NULL);
	sigaction(SIGILL, &sa, NULL);
	sigaction(SIGABRT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);

	std::shared_ptr<tbox::InputDatabase> input_db(new tbox::InputDatabase("input_db"));
	std::string input_file(argv[1]);
	tbox::InputManager::getManager()->parseInputFile(input_file, input_db);
	if (!input_db->keyExists("Main")) return -1;
	std::shared_ptr<tbox::Database> main_db(input_db->getDatabase("Main"));
	dt = main_db->getDouble("dt");
	bool rebalance_mesh = main_db->getBoolWithDefault("rebalance_processors", false);

	const tbox::Dimension dim(static_cast<unsigned short>(3));

	// Get the restart manager and root restart database.If run is from restart, open the restart file.
	is_from_restart = main_db->getBool("start_from_restart");
	int restart_interval = main_db->getInteger("restart_interval");
	string restart_write_dirname = main_db->getStringWithDefault("restart_dirname", ".");
	const bool write_restart = (restart_interval > 0) && !(restart_write_dirname.empty());

	tbox::RestartManager* restart_manager = tbox::RestartManager::getManager();

	if (is_from_restart) {
		restore_num = main_db->getInteger("restart_iteration");
		restart_manager->openRestartFile(restart_write_dirname, restore_num, mpi.getSize());
	}

	MainRestartData* main_restart_data = new MainRestartData("MainRestartData",main_db);

	loop_time = main_restart_data->getLoopTime();
	int loop_cycle = main_restart_data->getIterationNumber();
	iteration_num = main_restart_data->getIterationNumber();

	//Setting the timers
	tbox::TimerManager::createManager(input_db->getDatabase("TimerManager"));
	std::shared_ptr<tbox::Timer> t_regrid(tbox::TimerManager::getManager()->getTimer("Regridding"));
	
	//Setup the hdf5 outputs
	int visit_number_procs_per_file = 1;
	std::shared_ptr<tbox::Database> writer_db(input_db->getDatabase("FileWriter"));
	//Full dump mesh
	if (writer_db->isDatabase("full_dump_mesh")) {
		std::shared_ptr<tbox::Database> full_writer_db(writer_db->getDatabase("full_dump_mesh"));
		//Evolution dump
		if (full_writer_db->keyExists("hdf5_dump_interval")) {
			viz_mesh_dump_interval = full_writer_db->getInteger("hdf5_dump_interval");
		}
		if ( viz_mesh_dump_interval > 0) {
			if (full_writer_db->keyExists("hdf5_dump_dirname")) {
				viz_mesh_dump_dirname = full_writer_db->getStringWithDefault("hdf5_dump_dirname", ".");
			} else {
				//The directory for the hdf5 output parameter does not exist
				return -1;
			}
			if (full_writer_db->keyExists("variables")) {
				full_mesh_writer_variables = full_writer_db->getStringVector("variables");
			} else {
				//The directory for the hdf5 output parameter does not exist
				return -1;
			}
			output_information << "    Full mesh domain output  " << endl;
			output_information << "    Output directory:  " << viz_mesh_dump_dirname << endl;
			output_information << "    Snapshot interval: " << viz_mesh_dump_interval << endl;
			output_information << "|--------------------------------------------------------|" << endl;
		}
	}


	//Integrals
	n_at_commands = static_cast<int>(writer_db->getAllKeys().size());
	for (int i = 0; i < n_at_commands; ++i) {
		std::string at_name = "integration_" + tbox::Utilities::intToString(i);
		if (writer_db->keyExists(at_name)) {
			std::shared_ptr<tbox::Database> integral_db(writer_db->getDatabase(at_name));
			string integral_dirname;
			if (integral_db->keyExists("ascii_dump_dirname")) {
				integral_dirname = integral_db->getStringWithDefault("ascii_dump_dirname", ".");
			} else {
				std::cerr << "Error in parameter file: Integration output must have parameter 'ascii_dump_dirname'." << endl;
				return -1;
			}
			std::vector<std::string> calculation(integral_db->getStringVector("calculation"));

			std::shared_ptr<IntegrateDataWriter> integral(new IntegrateDataWriter(calculation, at_name, integral_dirname));
			integrateDataWriters.push_back(integral);
			int integralInterval = 0;
			if (integral_db->keyExists("ascii_dump_interval")) {
				integralInterval = integral_db->getInteger("ascii_dump_interval");
				integralIntervals.push_back(integralInterval);
			} else {
				std::cerr << "Error in parameter file: Integration output must have parameter 'ascii_dump_interval'." << endl;
				return -1;
			}
			if (integral_db->keyExists("variables")) {
				vector<string> tmp = integral_db->getStringVector("variables");
				integralVariables.push_back(set<string>(tmp.begin(), tmp.end()));
			} else {
				std::cerr << "Error in parameter file: Integration output must have parameter 'variables'." << endl;
				return -1;
			}
			if (integralInterval > 0) {
				std::string result;
				for (std::vector<string>::iterator it = calculation.begin(); it != calculation.end(); ++it) {
					result += (*it) + " ";
				}
				output_information << "    Integration output  " << endl;
				output_information << "    Output directory:  " << integral_dirname << endl;
				output_information << "    Snapshot interval: " << integralInterval << endl;
				output_information << "    Calculation: " << result << endl;
				output_information << "|--------------------------------------------------------|" << endl;
			}
			integralAnalysis.push_back(integral_db->getBool("activate_analysis"));
		} else {
			//End loop when no more integral_x found
			break;
		}
	}
	//Points
	n_at_commands = static_cast<int>(writer_db->getAllKeys().size());
	for (int i = 0; i < n_at_commands; ++i) {
		std::string at_name = "point_" + tbox::Utilities::intToString(i);
		if (writer_db->keyExists(at_name)) {
			std::shared_ptr<tbox::Database> point_db(writer_db->getDatabase(at_name));
			string point_dirname;
			if (point_db->keyExists("ascii_dump_dirname")) {
				point_dirname = point_db->getStringWithDefault("ascii_dump_dirname", ".");
			} else {
				std::cerr << "Error in parameter file: Point output must have parameter 'ascii_dump_dirname'." << endl;
				return -1;
			}
			std::vector<double> coordinates(point_db->getDoubleVector("coordinates"));

			std::shared_ptr<PointDataWriter> point(new PointDataWriter(coordinates, at_name, point_dirname));
			pointDataWriters.push_back(point);
			int pointInterval = 0;
			if (point_db->keyExists("ascii_dump_interval")) {
				pointInterval = point_db->getInteger("ascii_dump_interval");
				pointIntervals.push_back(pointInterval);
			} else {
				std::cerr << "Error in parameter file: Point output must have parameter 'ascii_dump_interval'." << endl;
				return -1;
			}
			if (point_db->keyExists("variables")) {
				vector<string> tmp = point_db->getStringVector("variables");
				pointVariables.push_back(set<string>(tmp.begin(), tmp.end()));
			} else {
				std::cerr << "Error in parameter file: Point output must have parameter 'variables'." << endl;
				return -1;
			}
			if (pointInterval > 0) {
				std::stringstream ss;
				for(size_t i = 0; i < coordinates.size(); ++i) {
					if(i != 0)
						ss << ",";
					ss << coordinates[i];
				}
				std::string result = ss.str();
				output_information << "    Point output  " << endl;
				output_information << "    Output directory:  " << point_dirname << endl;
				output_information << "    Snapshot interval: " << pointInterval << endl;
				output_information << "    Coordinates: " << result << endl;
				output_information << "|--------------------------------------------------------|" << endl;
			}
			pointAnalysis.push_back(point_db->getBool("activate_analysis"));
		} else {
			//End loop when no more point_x found
			break;
		}
	}

	//Slices
	for (int i = 0; i < n_at_commands; ++i) {
		std::string at_name = "slice_" + tbox::Utilities::intToString(i);
		if (writer_db->keyExists(at_name)) {
			std::shared_ptr<tbox::Database> slicer_db(writer_db->getDatabase(at_name));
			string slicer_dirname;
			if (slicer_db->keyExists("hdf5_dump_dirname")) {
				slicer_dirname = slicer_db->getStringWithDefault("hdf5_dump_dirname", ".");
			} else {
				std::cerr << "Error in parameter file: Slice output must have parameter 'hdf5_dump_dirname'." << endl;
				return -1;
			}
			int plane_normal_axis;
			if (slicer_db->keyExists("plane_normal_axis")) {
				plane_normal_axis = slicer_db->getInteger("plane_normal_axis");
			} else {
				std::cerr << "Error in parameter file: Slice output must have parameter 'plane_normal_axis'." << endl;
				return -1;
			}
			double distance_to_origin;
			if (slicer_db->keyExists("distance_to_origin")) {
				distance_to_origin = slicer_db->getFloat("distance_to_origin");
			} else {
				std::cerr << "Error in parameter file: Slice output must have parameter 'distance_to_origin'." << endl;
				return -1;
			}
			std::shared_ptr<SlicerDataWriter> slicer(new SlicerDataWriter(at_name, slicer_dirname, plane_normal_axis, distance_to_origin, visit_number_procs_per_file));
			sliceWriters.push_back(slicer);
			int slicerInterval = 0;
			if (slicer_db->keyExists("hdf5_dump_interval")) {
				slicerInterval = slicer_db->getInteger("hdf5_dump_interval");
				sliceIntervals.push_back(slicerInterval);
			} else {
				std::cerr << "Error in parameter file: Slice output must have parameter 'hdf5_dump_interval'." << endl;
				return -1;
			}
			if (slicer_db->keyExists("variables")) {
				vector<string> tmp = slicer_db->getStringVector("variables");
				sliceVariables.push_back(set<string>(tmp.begin(), tmp.end()));
			} else {
				std::cerr << "Error in parameter file: Slice output must have parameter 'variables'." << endl;
				return -1;
			}
			if (slicerInterval > 0) {
				output_information << "    Slicer output  " << endl;
				output_information << "    Output directory:  " << slicer_dirname << endl;
				output_information << "    Snapshot interval: " << slicerInterval << endl;
				output_information << "    Plane normal axis: " << plane_normal_axis << endl;
				output_information << "    Distance to origin: " << distance_to_origin << endl;
				output_information << "|--------------------------------------------------------|" << endl;
			}
			sliceAnalysis.push_back(slicer_db->getBool("activate_analysis"));
		} else {
			//End loop when no more slice_x found
			break;
		}
	}
	//Spheres
	for (int i = 0; i < n_at_commands; ++i) {
		std::string at_name = "sphere_" + tbox::Utilities::intToString(i);
		if (writer_db->keyExists(at_name)) {
			std::shared_ptr<tbox::Database> sphere_db(writer_db->getDatabase(at_name));
			string sphere_dirname;
			if (sphere_db->keyExists("hdf5_dump_dirname")) {
				sphere_dirname = sphere_db->getStringWithDefault("hdf5_dump_dirname", ".");
			} else {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'hdf5_dump_dirname'." << endl;
				return -1;
			}
			double radius;
			if (sphere_db->keyExists("radius")) {
				radius = sphere_db->getDouble("radius");
			} else {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'radius'." << endl;
				return -1;
			}
			if (!sphere_db->keyExists("center")) {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'center'." << endl;
				return -1;
			}
			std::vector<double> center(sphere_db->getDoubleVector("center"));
			if(center.size() != 3) {
				std::cerr << "Error in parameter file: Sphere center should be composed by three values. (e.g. 0.0, 2.5, 4.7)" << endl;
				return -1;
			}
			if (!sphere_db->keyExists("resolution")) {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'resolution'." << endl;
				return -1;
			}
			std::vector<int> resolution(sphere_db->getIntegerVector("resolution"));
			if(resolution.size() != 2) {
				std::cerr << "Error in parameter file: Spherical resolution should be composed by two values. (e.g. 30, 40)" << endl;
				return -1;
			}
			std::shared_ptr<SphereDataWriter> sphere(new SphereDataWriter(at_name, sphere_dirname, radius, center, resolution));
			sphereWriters.push_back(sphere);
			int sphereInterval = 0;
			if (sphere_db->keyExists("hdf5_dump_interval")) {
				sphereInterval = sphere_db->getInteger("hdf5_dump_interval");
				sphereIntervals.push_back(sphereInterval);
			} else {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'hdf5_dump_interval'." << endl;
				return -1;
			}
			if (sphere_db->keyExists("variables")) {
				vector<string> tmp = sphere_db->getStringVector("variables");
				sphereVariables.push_back(set<string>(tmp.begin(), tmp.end()));
			} else {
				std::cerr << "Error in parameter file: Spherical output must have parameter 'variables'." << endl;
				return -1;
			}
			if (sphereInterval > 0) {
				std::string result;
				for (std::vector<double>::iterator it = center.begin(); it != center.end(); ++it) {
					std::ostringstream strs;
					strs << (*it);
					result += strs.str() + " ";
				}
				output_information << "    Spherical output  " << endl;
				output_information << "    Output directory:  " << sphere_dirname << endl;
				output_information << "    Snapshot interval: " << sphereInterval << endl;
				output_information << "    Center: " << result << endl;
				output_information << "    Radius: " << radius << endl;
				output_information << "    Resolution: " << resolution[0] << " " << resolution[1] << endl;
				output_information << "|--------------------------------------------------------|" << endl;
			}
			sphereAnalysis.push_back(sphere_db->getBool("activate_analysis"));
		} else {
			//End loop when no more sphere_x found
			break;
		}
	}
	




	std::shared_ptr<appu::VisItDataWriter > visit_data_writer(new appu::VisItDataWriter(dim,"Mesh VisIt Writer", viz_mesh_dump_dirname));


	//Setup the output
	output_interval = 0;
	if (main_db->keyExists("output_interval")) {
		output_interval = main_db->getInteger("output_interval");
	}
	timer_output_interval = 0;
	if (main_db->keyExists("timer_output_interval")) {
		timer_output_interval = main_db->getInteger("timer_output_interval");
	}

	//Mesh creation
	std::shared_ptr<geom::CartesianGridGeometry > grid_geometry(new geom::CartesianGridGeometry(dim,"CartesianGeometry", input_db->getDatabase("CartesianGeometry")));
	std::shared_ptr<hier::PatchHierarchy > patch_hierarchy(new hier::PatchHierarchy("PatchHierarchy", grid_geometry, input_db->getDatabase("PatchHierarchy")));
	//Refinement domain to index conversion
	std::shared_ptr<tbox::Database> sti_db = input_db->getDatabase("StandardTagAndInitialize");
	n_at_commands = static_cast<int>(sti_db->getAllKeys().size());
	for (int i = 0; i < n_at_commands; ++i) {
		std::string at_name = "at_" + tbox::Utilities::intToString(i);
		std::shared_ptr<tbox::Database> at_db(sti_db->getDatabase(at_name));
		int n_tag_keys = static_cast<int>(at_db->getAllKeys().size()) - 1;
		for (int j = 0; j < n_tag_keys; ++j) {
			std::string tag_name = "tag_" + tbox::Utilities::intToString(j);
			std::shared_ptr<tbox::Database> this_tag_db(at_db->getDatabase(tag_name));
			std::string tagging_method = this_tag_db->getString("tagging_method");

			if (tagging_method == "REFINE_BOXES") {
				std::vector<std::string> level_keys = this_tag_db->getAllKeys();
				int n_level_keys = static_cast<int>(level_keys.size());

				// For each level specified, read the refine boxes.
				int current_ratio_x = 1;
				int current_ratio_y = 1;
				int current_ratio_z = 1;
				for (int k = 0; k < n_level_keys; ++k) {
					hier::BoxContainer level_boxes;
					if (level_keys[k] == "tagging_method") {
						continue;
					}
					int level = atoi(level_keys[k].substr(6).c_str());
					if (level < patch_hierarchy->getMaxNumberOfLevels()) {
						current_ratio_x = current_ratio_x * patch_hierarchy->getRatioToCoarserLevel(level)[0];
						current_ratio_y = current_ratio_y * patch_hierarchy->getRatioToCoarserLevel(level)[1];
						if (dim == tbox::Dimension(3)) {
							current_ratio_z = current_ratio_z * patch_hierarchy->getRatioToCoarserLevel(level)[2];
						}
						std::shared_ptr<tbox::Database> level_db(this_tag_db->getDatabase(level_keys[k]));
						std::vector<std::string> box_keys =level_db->getAllKeys();
						int n_box_keys = static_cast<int>(box_keys.size());
						std::vector<tbox::DatabaseBox> box_vector;
						for (int l = 0; l < n_box_keys; ++l) {
							std::string box_name = "box_" + tbox::Utilities::intToString(l);
							std::shared_ptr<tbox::Database> box_db(level_db->getDatabase(box_name));
							std::vector<double> lower(box_db->getDoubleVector("x_lo"));
							std::vector<double> upper(box_db->getDoubleVector("x_up"));

							int* low = new int[dim.getValue()];
							int* up = new int[dim.getValue()];
							low[0] = floor((lower[0] - grid_geometry->getXLower()[0]) / (grid_geometry->getDx()[0]/current_ratio_x));
							low[1] = floor((lower[1] - grid_geometry->getXLower()[1]) / (grid_geometry->getDx()[1]/current_ratio_y));
							if (dim == tbox::Dimension(3)) {
								low[2] = floor((lower[2] - grid_geometry->getXLower()[2]) / (grid_geometry->getDx()[2]/current_ratio_z));
							}

							up[0] = ceil((upper[0] - grid_geometry->getXLower()[0]) / (grid_geometry->getDx()[0]/current_ratio_x)) - 1;
							up[1] = ceil((upper[1] - grid_geometry->getXLower()[1]) / (grid_geometry->getDx()[1]/current_ratio_y)) - 1;
							if (dim == tbox::Dimension(3)) {
								up[2] = ceil((upper[2] - grid_geometry->getXLower()[2]) / (grid_geometry->getDx()[2]/current_ratio_z)) - 1;
							}

							tbox::DatabaseBox box(dim, low, up);
							box_vector.push_back(box);
							delete[] low;
							delete[] up;
						}
						this_tag_db->putDatabase(level_keys[k]);
						std::shared_ptr<tbox::Database> new_level_db(this_tag_db->getDatabase(level_keys[k]));
						new_level_db->putDatabaseBoxVector("boxes", box_vector);
					}
					else {
						std::vector<tbox::DatabaseBox> box_vector;
						int* low = new int[dim.getValue()];
						low[0] = 0;
						low[1] = 0;
						low[2] = 0;
						tbox::DatabaseBox box(dim, low, low);
						box_vector.push_back(box);
						this_tag_db->putDatabase(level_keys[k]);
						std::shared_ptr<tbox::Database> new_level_db(this_tag_db->getDatabase(level_keys[k]));
						new_level_db->putDatabaseBoxVector("boxes", box_vector);
						delete[] low;
					}
				}
			}
		}
	}

	std::string problem_name("Problem");
	std::shared_ptr<tbox::Database> problem_db(input_db->getDatabase("Problem"));
	problem = new Problem(problem_name, dim, problem_db, grid_geometry, patch_hierarchy, *main_restart_data, dt, is_from_restart, output_interval, timer_output_interval, viz_mesh_dump_interval, full_mesh_writer_variables, visit_data_writer, sliceIntervals, sliceVariables, sliceWriters, sphereIntervals, sphereVariables, sphereWriters, sliceAnalysis, sphereAnalysis, integralIntervals, integralVariables, integrateDataWriters, integralAnalysis, pointIntervals, pointVariables, pointDataWriters, pointAnalysis);
	std::shared_ptr<mesh::StandardTagAndInitialize > sti(new mesh::StandardTagAndInitialize("StandardTagAndInitialize", problem, sti_db));
	// Set up the clustering.
    const std::string clustering_type = main_db->getStringWithDefault("clustering_type", "BergerRigoutsos");
    std::shared_ptr<mesh::BoxGeneratorStrategy> box_generator;
	if (clustering_type == "BergerRigoutsos") {
        std::shared_ptr<tbox::Database> abr_db(input_db->getDatabase("BergerRigoutsos"));
        std::shared_ptr<mesh::BoxGeneratorStrategy> berger_rigoutsos(new mesh::BergerRigoutsos(dim, abr_db));
        box_generator = berger_rigoutsos;
    } else if (clustering_type == "TileClustering") {
    	std::shared_ptr<tbox::Database> tc_db(input_db->getDatabase("TileClustering"));
        std::shared_ptr<mesh::BoxGeneratorStrategy> tile_clustering(new mesh::TileClustering(dim, tc_db));
        box_generator = tile_clustering;
    }
    // Set up the load balancer.
    std::shared_ptr<mesh::LoadBalanceStrategy> load_balancer;
    std::shared_ptr<mesh::LoadBalanceStrategy> load_balancer0;
    const std::string partitioner_type = main_db->getStringWithDefault("partitioner_type", "TreeLoadBalancer");
    if (partitioner_type == "TreeLoadBalancer") {
        std::shared_ptr<mesh::TreeLoadBalancer> tree_load_balancer(new mesh::TreeLoadBalancer(dim, "mesh::TreeLoadBalancer", input_db->getDatabase("TreeLoadBalancer"), std::shared_ptr<tbox::RankTreeStrategy>(new tbox::BalancedDepthFirstTree)));
        tree_load_balancer->setSAMRAI_MPI(tbox::SAMRAI_MPI::getSAMRAIWorld());

        std::shared_ptr<mesh::TreeLoadBalancer> tree_load_balancer0(new mesh::TreeLoadBalancer(dim, "mesh::TreeLoadBalancer0", input_db->getDatabase("TreeLoadBalancer"), std::shared_ptr<tbox::RankTreeStrategy>(new tbox::BalancedDepthFirstTree)));
        tree_load_balancer0->setSAMRAI_MPI(tbox::SAMRAI_MPI::getSAMRAIWorld());

        load_balancer = tree_load_balancer;
        load_balancer0 = tree_load_balancer0;
    } else if (partitioner_type == "CascadePartitioner") {
        std::shared_ptr<mesh::CascadePartitioner> cascade_partitioner(new mesh::CascadePartitioner(dim, "mesh::CascadePartitioner", input_db->getDatabase("CascadePartitioner")));
        cascade_partitioner->setSAMRAI_MPI(tbox::SAMRAI_MPI::getSAMRAIWorld());

        std::shared_ptr<mesh::CascadePartitioner> cascade_partitioner0(new mesh::CascadePartitioner(dim, "mesh::CascadePartitioner0", input_db->getDatabase("CascadePartitioner")));
        cascade_partitioner0->setSAMRAI_MPI(tbox::SAMRAI_MPI::getSAMRAIWorld());

        load_balancer = cascade_partitioner;
        load_balancer0 = cascade_partitioner0;
    } else if (partitioner_type == "ChopAndPackLoadBalancer") {

        std::shared_ptr<mesh::ChopAndPackLoadBalancer> cap_load_balancer(new mesh::ChopAndPackLoadBalancer(dim, "mesh::ChopAndPackLoadBalancer", input_db->getDatabase("ChopAndPackLoadBalancer")));

        load_balancer = cap_load_balancer;
        std::shared_ptr<mesh::CascadePartitioner> cascade_partitioner0(new mesh::CascadePartitioner(dim, "mesh::CascadePartitioner0", input_db->getDatabase("CascadePartitioner")));
        cascade_partitioner0->setSAMRAI_MPI(tbox::SAMRAI_MPI::getSAMRAIWorld());
        load_balancer0 = cascade_partitioner0;
    }

	std::shared_ptr< mesh::GriddingAlgorithm > gridding_algorithm(new mesh::GriddingAlgorithm(patch_hierarchy, "GriddingAlgorithm", input_db->getDatabase("GriddingAlgorithm"), sti, box_generator, load_balancer, load_balancer0));
	if (!gridding_algorithm) return -1;

   // std::shared_ptr< algs::TimeRefinementLevelStrategy >  timeRefinementStrategy(SAMRAI_SHARED_PTR_CAST<algs::TimeRefinementLevelStrategy, Problem*>(problem));
    std::shared_ptr<algs::TimeRefinementLevelStrategy> timeRefinementStrategy(problem);
    std::shared_ptr<algs::TimeRefinementIntegrator> time_integrator(new algs::TimeRefinementIntegrator("TimeRefinementIntegrator", input_db->getDatabase("TimeRefinementIntegrator"), patch_hierarchy, timeRefinementStrategy, gridding_algorithm));



	//Prints the banner of the simulation
	if (my_proc == MPI_MASTER) {
		cout << "|-----------------------SIMPLAT--------------------------|" << endl;
		cout << "    Number of processors " << mpi.getSize()  << endl;
		cout << "|--------------------------------------------------------|" << endl;
		cout << output_information.str();
	}

	problem->setupPlotterMesh(*visit_data_writer);
	problem->setupSlicePlotter(sliceWriters);
	problem->setupSpherePlotter(sphereWriters);
	problem->setupIntegralPlotter(integrateDataWriters);
	problem->setupPointPlotter(pointDataWriters);


	//Hierarchy initialization
	double dt_now = time_integrator->initializeHierarchy();
	if (tbox::RestartManager::getManager()->isFromRestart() ) {
		//Allocates temporal mesh variables after a restart
		problem->allocateAfterRestart();
		//Rebalancing of mesh (in case more processors are needed)
		if (rebalance_mesh) {
			gridding_algorithm->makeCoarsestLevel(time_integrator->getIntegratorTime());
		}

	}
	tbox::RestartManager::getManager()->closeRestartFile();

	problem->postInit();

	//Print the initial data.

	if (viz_mesh_dump_interval > 0 && !tbox::RestartManager::getManager()->isFromRestart()) {
		visit_data_writer->writePlotData( patch_hierarchy, iteration_num, loop_time);
	}

	//Integral output
	if (integralIntervals.size() > 0 && !tbox::RestartManager::getManager()->isFromRestart()) {
		int i = 0;
		for (std::vector<std::shared_ptr<IntegrateDataWriter> >::iterator it = integrateDataWriters.begin(); it != integrateDataWriters.end(); ++it) {
			if (integralIntervals[i] > 0) {
				(*it)->writePlotData(patch_hierarchy, iteration_num, loop_time);
			}
			i++;
		}
	}
	//Point output
	if (pointIntervals.size() > 0 && !tbox::RestartManager::getManager()->isFromRestart()) {
		int i = 0;
		for (std::vector<std::shared_ptr<PointDataWriter> >::iterator it = pointDataWriters.begin(); it != pointDataWriters.end(); ++it) {
			if (pointIntervals[i] > 0) {
				(*it)->writePlotData(patch_hierarchy, iteration_num, loop_time);
			}
			i++;
		}
	}

	//Slice output
	if (sliceIntervals.size() > 0 && !tbox::RestartManager::getManager()->isFromRestart()) {
		int i = 0;
		for (std::vector<std::shared_ptr<SlicerDataWriter> >::iterator it = sliceWriters.begin(); it != sliceWriters.end(); ++it) {
			if (sliceIntervals[i] > 0) {
				(*it)->writePlotData(patch_hierarchy, iteration_num, loop_time);
			}
			i++;
		}
	}
	//Spherical output
	if (sphereIntervals.size() > 0 && !tbox::RestartManager::getManager()->isFromRestart()) {
		int i = 0;
		for (std::vector<std::shared_ptr<SphereDataWriter> >::iterator it = sphereWriters.begin(); it != sphereWriters.end(); ++it) {
			if (sphereIntervals[i] > 0) {
				(*it)->writePlotData(patch_hierarchy, iteration_num, loop_time);
			}
			i++;
		}
	}

	//Initial level information
	if (my_proc == MPI_MASTER) {
		cout << "|--------------------------------------------------------|" << endl;
		cout << "    Number of levels " << patch_hierarchy->getNumberOfLevels() << endl;
	}
	for (int ln = 0; ln < patch_hierarchy->getNumberOfLevels(); ln++) {
		std::shared_ptr< hier::PatchLevel > level(patch_hierarchy->getPatchLevel(ln));
		const std::shared_ptr<geom::CartesianGridGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianGridGeometry, hier::BaseGridGeometry>(level->getGridGeometry()));
		const double* deltas = patch_geom->getDx();
		const hier::IntVector ratio = level->getRatioToLevelZero();
		int numPatches = level->getGlobalNumberOfPatches();
		int numCells = level->getGlobalNumberOfCells();
		if (my_proc == MPI_MASTER) {
			cout<<"    Level: "<<ln<<". Number of patches: "<<numPatches<<". Number of cells: "<<numCells<<". Dx: "<<deltas[0]/ratio[0]<<" "<<deltas[1]/ratio[1]<<" "<<deltas[2]/ratio[2]
<<endl;
		}
	}
	if (my_proc == MPI_MASTER) {
		cout << "|--------------------------------------------------------|" << endl;
	}
	//Print memory info
	mpi.Barrier();
	tbox::MemoryUtilities::printMemoryInfo(cout);

	//Simulation steps
	double loop_time = time_integrator->getIntegratorTime();
	int iteration_num = time_integrator->getIntegratorStep();
	while (!problem->checkFinalization(loop_time, dt_now)) {
		//Do a step
		iteration_num = time_integrator->getIntegratorStep() + 1;

		problem->registerIteration(iteration_num);
		double dt_new = time_integrator->advanceHierarchy(dt_now);

		loop_time += dt_now;
		dt_now = dt_new;
		
		//Level information after hierarchy change
		if (time_integrator->atRegridPoint(0)) {
			for (int ln = 0; ln < patch_hierarchy->getNumberOfLevels(); ln++) {
				std::shared_ptr< hier::PatchLevel > level(patch_hierarchy->getPatchLevel(ln));
				const std::shared_ptr<geom::CartesianGridGeometry > patch_geom(SAMRAI_SHARED_PTR_CAST<geom::CartesianGridGeometry, hier::BaseGridGeometry>(level->getGridGeometry()));
				const double* deltas = patch_geom->getDx();
				const hier::IntVector ratio = level->getRatioToLevelZero();
				int numPatches = level->getGlobalNumberOfPatches();
				int numCells = level->getGlobalNumberOfCells();
				if (my_proc == MPI_MASTER) {
					cout<<"    Level: "<<ln<<". Number of patches: "<<numPatches<<". Number of cells: "<<numCells<<". Dx: "<<deltas[0]/ratio[0]<<" "<<deltas[1]/ratio[1]<<" "<<deltas[2]/ratio[2]
<<endl;
				}
			}
			if (my_proc == MPI_MASTER) {
				cout << "|--------------------------------------------------------|" << endl;
			}
			//Print memory info
			mpi.Barrier();
			tbox::MemoryUtilities::printMemoryInfo(cout);
		}


		//Output restart data
		if ( write_restart && (0 == iteration_num % restart_interval) ) {
			problem->putToRestart(*main_restart_data);
			tbox::RestartManager::getManager()->writeRestartFile(restart_write_dirname, iteration_num);
		}
	}
	//Last restart at the end of the simulation
	if ( write_restart) {
		problem->putToRestart(*main_restart_data);
		tbox::RestartManager::getManager()->writeRestartFile(restart_write_dirname, iteration_num);
	}

	//Print the end output
	if (my_proc == MPI_MASTER) {
		cout << "    Simulation finished:" << endl;
		cout << "           Iterations: " << iteration_num << endl;
		cout << "           Time: " << loop_time  << endl;	
		cout << "|--------------------------------------------------------|" << endl;

		tbox::TimerManager::getManager()->print(cout);
	}
	else {
		//Dispose other processor timers
		//SAMRAI needs all processors run tbox::TimerManager::getManager()->print, otherwise it hungs
		std::ofstream ofs;
		ofs.setstate(std::ios_base::badbit);
		tbox::TimerManager::getManager()->print(ofs);	
	}

	//Close all the objects
	time_integrator.reset();
	gridding_algorithm.reset();
	load_balancer.reset();
	box_generator.reset(); 
	sti.reset();
	if (main_restart_data) delete main_restart_data;

	patch_hierarchy.reset();
	grid_geometry.reset();
	input_db.reset();
	main_db.reset();
	tbox::SAMRAIManager::shutdown();
	tbox::SAMRAIManager::finalize();
	//Should be here, but an error is thrown if not commented.
	//tbox::SAMRAI_MPI::finalize();
	return 0;
}
