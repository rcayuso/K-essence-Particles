/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:	  (c) 1997-2016 Lawrence Livermore National Security, LLC
 * Description:	Linear time interp operator for node-centered double patch data.
 *
 ************************************************************************/
#include "TimeInterpolator.h"

#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/Index.h"
#include "SAMRAI/pdat/NodeData.h"
#include "SAMRAI/pdat/NodeVariable.h"
#include "SAMRAI/tbox/Utilities.h"
#include "SAMRAI/tbox/MathUtilities.h"
#include "SAMRAI/tbox/TimerManager.h"
#include "SAMRAI/pdat/IndexData.h"
#include "SAMRAI/pdat/CellGeometry.h"
#include "Commons.h"



#define vector3D(v, i, j, k) (v)[i+ilast*(j+jlast*(k))]
#define vector2D(v, i, j) (v)[i+ilast*(j)]

#define dvector3D(v, i, j, k) (v)[i+dilast*(j+djlast*(k))]
#define dvector2D(v, i, j) (v)[i+dilast*(j)]

#define greaterEq(a,b) ((fabs((a) - (b))/1.0E-9 > 10 ? false : (floor(fabs((a) - (b))/1.0E-9) < 1)) || (b)<(a))
#define lessEq(a,b) ((fabs((a) - (b))/1.0E-9 > 10 ? false: (floor(fabs((a) - (b))/1.0E-9) < 1)) || (a)<(b))
#define equalsEq(a,b) ((fabs((a) - (b))/1.0E-9 > 10 ? false: (floor(fabs((a) - (b))/1.0E-9) < 1)))

using namespace SAMRAI;

std::shared_ptr<tbox::Timer> TimeInterpolator::t_interpolate;
double TimeInterpolator::current_time;
double TimeInterpolator::interp_time;
double TimeInterpolator::new_time;

TimeInterpolator::TimeInterpolator(const std::shared_ptr<geom::CartesianGridGeometry >& grid_geom,
						  const std::string discType):
	hier::TimeInterpolateOperator(),
	d_discType(discType)
{
	t_interpolate = tbox::TimerManager::getManager()->getTimer("Time interpolate");

	xGlower = grid_geom->getXLower()[0];
	xGupper = grid_geom->getXUpper()[0];
	yGlower = grid_geom->getXLower()[1];
	yGupper = grid_geom->getXUpper()[1];
	zGlower = grid_geom->getXLower()[2];
	zGupper = grid_geom->getXUpper()[2];


	time_substep_number = 0;
}

TimeInterpolator::~TimeInterpolator()
{
}

inline void TimeInterpolator::interpolation(
	double simPlat_dt,
		double& un,
		double& rk1,
		double& rk2,
		double& rk3,
		double& unp1) const {
	double theta, k1, k2, k3, k4, ybase, yp, ypp, yppp, fypp;

	theta = time_substep_number / interp_ratio;
	k1 = 2.0 * (rk1 - un);
	k2 = 2.0 * (rk2 - un);
	k3 = rk3 - un;
	k4 = (6.0 * unp1 + 2.0 * un) - (2.0 * rk1 + 4.0 * rk2 + 2.0 * rk3);
	ybase = un + theta * k1 + 0.5 * theta * theta * ((-3.0) * k1 + 2.0 * k2 + 2.0 * k3 + (-k4)) + 2.0 / 3.0 * theta * theta * theta * ((k1 + k4) - (k2 + k3));
	yp = 1.0 / simPlat_dt * (k1 + theta * ((-3.0) * k1 + 2.0 * k2 + 2.0 * k3 + (-k4)) + 2.0 * theta * theta * ((k1 + k4) - (k2 + k3)));
	ypp = 1.0 / (simPlat_dt * simPlat_dt) * (((-3.0) * k1 + 2.0 * k2 + 2.0 * k3 + (-k4)) + 4.0 * theta * ((k1 + k4) - (k2 + k3)));
	yppp = 1.0 / (simPlat_dt * simPlat_dt * simPlat_dt) * (4.0 * ((k1 + k4) - (k2 + k3)));
	fypp = 4.0 / (simPlat_dt * simPlat_dt * simPlat_dt) * (k3 - k2);
	un = ybase;
	simPlat_dt = simPlat_dt / interp_ratio;
	k1 = simPlat_dt * yp;
	rk1 = ybase + k1 * 0.5;
	k2 = simPlat_dt * yp + simPlat_dt * simPlat_dt * ypp * 0.5 + simPlat_dt * simPlat_dt * simPlat_dt * (yppp - fypp) * 0.125;
	rk2 = ybase + k2 * 0.5;
	k3 = simPlat_dt * yp + simPlat_dt * simPlat_dt * ypp * 0.5 + simPlat_dt * simPlat_dt * simPlat_dt * (yppp + fypp) * 0.125;
	rk3 = ybase + k3;
	k4 = simPlat_dt * yp + simPlat_dt * simPlat_dt * ypp + simPlat_dt * simPlat_dt * simPlat_dt * yppp * 0.5;
	unp1 = ybase + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}


void TimeInterpolator::timeInterpolate(
	hier::PatchData& dst_data,
	const hier::Box& where,
	const hier::BoxOverlap& overlap,
	const hier::PatchData& src_data_un,
	const hier::PatchData& src_data_rk1,
	const hier::PatchData& src_data_rk2,
	const hier::PatchData& src_data_rk3,
	const hier::PatchData& src_data_unp1) const
{
	t_interpolate->start();
	const tbox::Dimension& dim(where.getDim());
	current_time = tbox::MathUtilities<double>::getMax();

	if (d_discType.compare("mesh") == 0) {
		const pdat::NodeData<double>* un_dat = CPP_CAST<const pdat::NodeData<double> *>(&src_data_un);
		const pdat::NodeData<double>* rk1_dat = CPP_CAST<const pdat::NodeData<double> *>(&src_data_rk1);
		const pdat::NodeData<double>* rk2_dat = CPP_CAST<const pdat::NodeData<double> *>(&src_data_rk2);
		const pdat::NodeData<double>* rk3_dat = CPP_CAST<const pdat::NodeData<double> *>(&src_data_rk3);
		const pdat::NodeData<double>* unp1_dat = CPP_CAST<const pdat::NodeData<double> *>(&src_data_unp1);
		pdat::NodeData<double>* dst_dat = CPP_CAST<pdat::NodeData<double> *>(&dst_data);

		TBOX_ASSERT(dst_dat != 0);
		TBOX_ASSERT((where * dst_dat->getGhostBox()).isSpatiallyEqual(where));

		const hier::Index ilo = un_dat->getGhostBox().lower();
		const hier::Index ihi = un_dat->getGhostBox().upper();
		const hier::Index dilo = dst_dat->getGhostBox().lower();
		const hier::Index dihi = dst_dat->getGhostBox().upper();

		const hier::Index ifirstf = where.lower();
		const hier::Index ilastf = where.upper();

		int ghosts = dst_data.getGhostCellWidth()[0];

		new_time = 0;
		current_time = tbox::MathUtilities<double>::Min(current_time, un_dat->getTime());
		new_time = tbox::MathUtilities<double>::Max(new_time, un_dat->getTime());
		current_time = tbox::MathUtilities<double>::Min(current_time, rk1_dat->getTime());
		new_time = tbox::MathUtilities<double>::Max(new_time, rk1_dat->getTime());
		current_time = tbox::MathUtilities<double>::Min(current_time, rk2_dat->getTime());
		new_time = tbox::MathUtilities<double>::Max(new_time, rk2_dat->getTime());
		current_time = tbox::MathUtilities<double>::Min(current_time, rk3_dat->getTime());
		new_time = tbox::MathUtilities<double>::Max(new_time, rk3_dat->getTime());
		current_time = tbox::MathUtilities<double>::Min(current_time, unp1_dat->getTime());
		new_time = tbox::MathUtilities<double>::Max(new_time, unp1_dat->getTime());
		const double dt = new_time - current_time;
		interp_time = dst_dat->getTime();

		TBOX_ASSERT(dt > 0);
		TBOX_ASSERT((current_time < interp_time ||
			tbox::MathUtilities<double>::equalEps(current_time, interp_time)) &&
			(interp_time < new_time ||
			tbox::MathUtilities<double>::equalEps(interp_time, new_time)));
		for (int d = 0; d < dst_dat->getDepth(); ++d) {
			double* field = dst_dat->getPointer(d);
			const double* field_un = un_dat->getPointer(d);
			const double* field_rk1 = rk1_dat->getPointer(d);
			const double* field_rk2 = rk2_dat->getPointer(d);
			const double* field_rk3 = rk3_dat->getPointer(d);
			const double* field_unp1 = unp1_dat->getPointer(d);

			int ilast = ihi(0)-ilo(0) + 2;
			int dilast = dihi(0)-dilo(0) + 2;
			int jlast = ihi(1)-ilo(1) + 2;
			int djlast = dihi(1)-dilo(1) + 2;

			for(int k = ifirstf[2] - ghosts; k <= ilastf[2] + ghosts; k++) {
				for(int j = ifirstf[1] - ghosts; j <= ilastf[1] + ghosts; j++) {
					for(int i = ifirstf[0] - ghosts; i <= ilastf[0] + ghosts; i++) {
						if (i >= dilo[0] && i <= dihi[0] && j >= dilo[1] && j <= dihi[1] && k >= dilo[2] && k <= dihi[2]) {
							double simPlat_dt = dt;
							double un = vector3D(field_un, i - ilo[0], j - ilo[1], k - ilo[2]);
							double rk1 = vector3D(field_rk1, i - ilo[0], j - ilo[1], k - ilo[2]);
							double rk2 = vector3D(field_rk2, i - ilo[0], j - ilo[1], k - ilo[2]);
							double rk3 = vector3D(field_rk3, i - ilo[0], j - ilo[1], k - ilo[2]);
							double unp1 = vector3D(field_unp1, i - ilo[0], j - ilo[1], k - ilo[2]);
							interpolation(simPlat_dt, un, rk1, rk2, rk3, unp1);
							if (interp_step == 0) {
								dvector3D(field, i - dilo[0], j - dilo[1], k - dilo[2]) = un;
							}
							if (interp_step == 1) {
								dvector3D(field, i - dilo[0], j - dilo[1], k - dilo[2]) = rk1;
							}
							if (interp_step == 2) {
								dvector3D(field, i - dilo[0], j - dilo[1], k - dilo[2]) = rk2;
							}
							if (interp_step == 3) {
								dvector3D(field, i - dilo[0], j - dilo[1], k - dilo[2]) = rk3;
							}
							if (interp_step == 4) {
								dvector3D(field, i - dilo[0], j - dilo[1], k - dilo[2]) = unp1;
							}
						}
					}
				}
			}
		}
	}
	t_interpolate->stop();
}



