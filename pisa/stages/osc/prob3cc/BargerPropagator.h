#ifndef __BargerPropagator__
#define __BargerPropagator__

#include "EarthDensity.h"
#include "NeutrinoPropagator.h"

#include "mosc3.h"
#include "mosc.h"

#include <iostream>
#include <stdlib.h>
#include <string.h>

// A neutrino propagator class based on the 1980 Barger paper
// PRD.22.11, Dec. 1 1980.
//
// The underlying oscillation code is written in mosc*
//
// Capable of computing oscillation probabilities through constant density
// matter or through spheres of varying radial density.

class BargerPropagator : public NeutrinoPropagator
{
  public:
      BargerPropagator( );
	  // const char specifies an input radial density profile, c.f. PREM.dat
      BargerPropagator( bool );
      BargerPropagator( const char *, double detectorDepth = 0.0 );
     ~BargerPropagator( );

      // Main driving routine for computing oscillations through a sphere,
      // to be called after SetMNS(...).
	  //
      // specify neutrino type:   +int : neutrino   -int: anti-neutrino
      virtual void propagate( int );

	  // Driving routine for oscillations through linear media of constant
	  // density, to be called after SetMNS(...).
	  //
      // specify:
	  // 	neutrino type:   +int : neutrino   -int: anti-neutrino
      // 	path length in the matter
      // 	density of the matter
      virtual void propagateLinear( int , double, double );

      // Driving routine for oscillations in vacuum.
      // called after SetMNS(...)
      // Args:                    |nu_in| nu_out| Energy | pathlength
      //                          |     |       | (GeV)  | (km)
      virtual double GetVacuumProb( int , int   , double , double );

	  // determines the pathlength and density profile for a neutrino
	  // propagating through a sphere
	  //
	  // specify:
	  // 	cosine of zenith angle:
	  // 		-1=upward going, 0=horizontal, +1=downward going
	  //
      // 	production height in the atmosphere (km)
	  //
	  // 	if the profile withing EarthDensity object should be recomputed,
	  // 		default is true
      virtual void DefinePath( double, double,
                               double YeI, double YeO, double YeM,
                               bool kSetProfile = true );

      // Determine the neutrino oscillation parameters.
	  //
      // This routine must be called _before_ propagate* routines!
	  //
	  // Specify the neutrino oscillation parameters, energy, a bool for which
	  // form of mixing angle is input, and the last argument is the neutrino
	  // type (nu > 0 : neutrinos  nu < 0 : antineutrinos)
	  //
	  // This type must agree with the type used in the call to propagate() and
	  // propagateLinear()
	  // TODO: Why would you specify the neutrino type thrice, where any one
	  //       being different will break the result???
	  //
      // Args:           |  x12   |  x13   |  x23   |  dm21  |  dm32  |  d_cp  | Energy |T:s^2(x) |1:nu
      //                 |        |        |        | (eV^2) | (eV^2) | (rad)  | (GeV)  |F:s^2(2x)|2:antinu
      virtual void SetMNS( double , double , double , double , double , double , double , bool    , int );

      // Change the conversion factor from matter density to electron density.
      void SetDensityConversion( double x ) { density_convert = x; }

      // Return oscillation probabilities nu_in -> nu_out
	  //
      // Values for nuIn and nuOut by flavor are:
	  // 	0=nue{bar}, 1=numu{bar}, 2=nutau{bar}
	  // where whether neutrinos or antineutrinos are chosen is based upon how
	  // the calculation was set up.
      double GetProb( int nuIn, int nuOut ){
         return Probability[nuIn][nuOut];
      };

      // Miscellaneous
      double GetPathLength( ){ return Earth->get_Pathlength(); }
      void SetPathLength( double x ){ PathLength = x; }
      void SetEnergy( double x ){ Energy = x; }
      virtual void SetMatterPathLength( );
      virtual void SetAirPathLength(double);


	  // Specify weather oscillation probabilities are computed from neutrino
	  // mass eigenstates of from neutrino flavor eigenstates
	  // 	true = mass eigenstates, false = flavor eigenstates
      void UseMassEigenstates( bool x ) { kUseMassEigenstates = x; }

      void SetWarningSuppression( bool x = true ) { kSuppressWarnings = x; }

      // Specify how the user inputs the atmospheric neutrino mixing mass.
	  //
      // true (default mode) means the mixing input for SetMNS corresponds to
      //    NH: m32
      //    IH: m31
      //  That is, in this mode the code will correct the input value of the
      //  atmospheric mass splitting parameter by the solar mass splitting if
      //  the input is negative (corresponding to IH input).
      //  If SetOneMassScaleMode(false) is called this correction is not
      //  performed, and the user is responsible for supplying the correct
      //  value of Dm23 for oscillations in both hierarchies.
      void SetOneMassScaleMode( bool x = true )  { kOneDominantMass = x; }

      virtual void fill_osc_prob_c(double *ecen, int ecen_length, double *czcen, 
        int czcen_length, double energy_scale, double deltam21, double deltam31,
        double deltacp, double prop_height, double YeI, double YeO, double YeM,
        double *probList, int prop_length, double *evals, int evals_length, 
        double *czvals, int czvals_length, double theta12, double theta13, 
        double theta23);
      
  protected:
      void init( );

      void ClearProbabilities( );

      double Probability[3][3];

      EarthDensity * Earth;

      double REarth;
      double RDetector;
      double ProductionHeight;
      double PathLength;
      double AirPathLength;
      double MatterPathLength;
      double CosineZenith;
      double Energy;
      double density_convert;


      bool kAntiMNSMatrix;
      bool kSuppressWarnings;
      bool kUseMassEigenstates;
      bool kOneDominantMass;
};

#endif