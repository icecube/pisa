#include "BargerPropagator.h"

BargerPropagator::BargerPropagator()
{
   Earth = new EarthDensity( );
   init();
}


BargerPropagator::BargerPropagator( bool k )
{
   Earth = new EarthDensity( );
   init();
}


BargerPropagator::~BargerPropagator( )
{
   delete Earth;
}

BargerPropagator::BargerPropagator( const char * f, double detectorDepth )
{
  Earth = new EarthDensity( f, detectorDepth );
  init();
}

void BargerPropagator::init()
{
   kUseMassEigenstates = false;

   //rad earth in [cm] /
   //REarth = Earth->GetEarthRadiuskm() * 1.0e5;
   RDetector = Earth->get_RDetector() * 1.0e5;
   ProductionHeight = 0.0;
   PathLength = 0.0;

   // default is neutral matter
   density_convert = 0.5;

   kAntiMNSMatrix     = false ;
   kSuppressWarnings  = false ;

   kOneDominantMass   = true  ;
}



void BargerPropagator::propagate( int NuFlavor ){

   int    i,j;
   int    Layers;
   double TransitionMatrix[3][3][2];
   double TransitionProduct[3][3][2];
   double TransitionTemp[3][3][2];
   double RawInputPsi[3][2];
   double OutputPsi[3][2];


   if( ! kSuppressWarnings )
   if(
       ( kAntiMNSMatrix && NuFlavor > 0) ||
       (!kAntiMNSMatrix && NuFlavor < 0)
     )
   {
     std::cout << " Warning BargerPropagator::propagate - " << std::endl;
     std::cout << "     Propagating neutrino flavor and MNS matrix definition differ :" << std::endl;
      std::cout << "     MNS Matrix was defined for : " << ( kAntiMNSMatrix ? " Nubar " : "Nu" )<< std::endl;
      std::cout << "     Propagation is for         : " << ( NuFlavor < 0   ? " Nubar " : "Nu" )<< std::endl;
      std::cout << "     Please check your call to BargerPropagator::SetMNS() " << std::endl;
      std::cout << "     This message can be suppressed with a call to BargerPropagator::SuppressWarnings() " << std::endl;
      exit(-1);
   }

   clear_complex_matrix( TransitionMatrix );
   clear_complex_matrix( TransitionProduct );
   clear_complex_matrix( TransitionTemp );

   ClearProbabilities();


   Earth->SetDensityProfile( CosineZenith, PathLength, ProductionHeight );
   Layers = Earth->get_LayersTraversed( );

   //cout<<"density, distanceAcrossLayer, density_convert:"<<endl;

   // TCA: (07May2014) Modifying this next for loop
   // Adding electron fraction to be something other than 0.5 in each layer
   for ( i = 0; i < Layers ; i++ ) {
     double density = Earth->get_DensityInLayer(i);
     double distanceAcrossLayer = Earth->get_DistanceAcrossLayer(i)/1.0e5;
     density_convert = Earth->get_ElectronFractionInLayer(i);

     //cout<<density<<"   "<<distanceAcrossLayer<<"    "<<density_convert<<endl;

     get_transition_matrix( NuFlavor,
			    Energy,		         // in GeV
			    density*density_convert,
			    distanceAcrossLayer,         // in km
			    TransitionMatrix,	         // Output transition matrix
			    0.0  			 // phase offset
			    );

     if ( i == 0 )
       copy_complex_matrix( TransitionMatrix , TransitionProduct );

      if ( i >0 ){
         clear_complex_matrix( TransitionTemp );
         multiply_complex_matrix( TransitionMatrix, TransitionProduct, TransitionTemp );
         copy_complex_matrix( TransitionTemp, TransitionProduct );
      }//for other layers
    }// end of layer loop


   // loop on neutrino types
   for ( i = 0 ; i < 3 ; i++ )
   {
      for ( j = 0 ; j < 3 ; j++ )
      {  RawInputPsi[j][0] = 0.0; RawInputPsi[j][1] = 0.0;   }

      if( kUseMassEigenstates )
         convert_from_mass_eigenstate( i+1, NuFlavor,  RawInputPsi );
      else
         RawInputPsi[i][0] = 1.0;


      multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );
      Probability[i][0] += OutputPsi[0][0] * OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
      Probability[i][1] += OutputPsi[1][0] * OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
      Probability[i][2] += OutputPsi[2][0] * OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];

   }//end of neutrino loop

}




void BargerPropagator::ClearProbabilities()
{
   for ( int i = 0 ; i < 3; i++ )
      for ( int j = 0 ; j < 3 ; j++ )
         Probability[i][j] = 0.0;

}

void BargerPropagator::SetMNS( double x12, double x13, double x23,
                               double m21, double mAtm, double delta,
                               double Energy_ , bool kSquared, int kNuType )
{
   Energy = Energy_;

   double sin12;
   double sin13;
   double sin23;

   double lm32 = mAtm ;
   // Dominant Mixing mode assumes the user
   // simply changes the sign of the input atmospheric
   // mixing to invert the hierarchy
   //  so the input for  NH corresponds to m32
   // and the input for  IH corresponds to m31
   if( kOneDominantMass )
   {
      // For the inverted Hierarchy, adjust the input
      // by the solar mixing (should be positive)
      // to feed the core libraries the correct value of m32
      if( mAtm < 0.0 ) lm32 = mAtm - m21 ;
   }
   else
   {
       if( !kSuppressWarnings )
       {
         std::cout << " BargerPropagator::SetMNS - " << std::endl;
         std::cout << "     You have opted to specify the value of m23 by yourself. " << std::endl;
         std::cout << "     This means you must correct the value of m23 when switching " << std::endl;
         std::cout << "     between the mass hierarchy options. " << std::endl;
         std::cout << "     This message can be suppressed with BargerPropagator::SuppressWarnings()"<< std::endl;
      }
   }



   //if xAB = sin( xAB )^2
   if ( kSquared ){
      sin12 = sqrt( x12 );
      sin13 = sqrt( x13 );
      sin23 = sqrt( x23 );
   }
   else
   {
      //if xAB = sin( 2 xAB )^2
      sin12 = sqrt( 0.5*(1 - sqrt(1 - x12 ))  );
      sin13 = sqrt( 0.5*(1 - sqrt(1 - x13 ))  );
      sin23 = sqrt( 0.5*(1 - sqrt(1 - x23 ))  );
   }

   if ( kNuType < 0 )
   {
     delta *= -1.0 ;
     kAntiMNSMatrix = true ;
   }
   else
   {
     kAntiMNSMatrix = false ;
   }

   init_mixing_matrix( m21, lm32, sin12, sin23, sin13, delta );

}

void BargerPropagator::DefinePath(double cz, double ProdHeight,
                                  double YeI, double YeO, double YeM,
                                  bool kSetProfile )
/*
  10May2014 (TCA): Changed this to calculate the correct PathLength
  taking into account detector depth and a neutrino path above the
  horizon.
*/
{

  double depth = Earth->get_DetectorDepth()*1.0e5;
  //cout<<"In DefinePath(): depth: "<<depth<<endl;

  ProductionHeight = ProdHeight*1e5;

  if(cz < 0) {
    PathLength = sqrt((RDetector + ProductionHeight +depth)*(RDetector + ProductionHeight +depth)
		      - (RDetector*RDetector)*( 1 - cz*cz)) - RDetector*cz;
  } else {
    double kappa = (depth + ProductionHeight)/RDetector;
    PathLength = RDetector*sqrt(cz*cz - 1 + (1 + kappa)*(1 + kappa)) - RDetector*cz;

  }


  CosineZenith = cz;
  if( kSetProfile ) {
    Earth->SetDensityProfile( CosineZenith, PathLength, ProductionHeight );
    Earth->SetElecFrac(YeI, YeO, YeM);
  }
}


void BargerPropagator::SetMatterPathLength()
{

   int Layers = Earth->get_LayersTraversed( );

   MatterPathLength = 0.0;
   AirPathLength = 0.0;
   for( int i = 1 ; i < Layers ; i++ )
      MatterPathLength +=  Earth->get_DistanceAcrossLayer(i);

   AirPathLength +=  Earth->get_DistanceAcrossLayer(0);
}

void BargerPropagator::SetAirPathLength(double x)
{
//      argument is [km], convert to [cm]
      AirPathLength = x*1.0e5 - MatterPathLength;
}



double BargerPropagator::GetVacuumProb( int Alpha, int Beta , double Energy, double Path )
{
   // alpha -> 1:e 2:mu 3:tau
   // Energy[GeV]
   // Path[km]
   /// simple referes to the fact that in the 3 flavor analysis
   //  the solar mass term is zero
   double Probs[3][3];


   get_vacuum_probability( Alpha, Energy, Path, Probs );

   Alpha = abs(Alpha);
   Beta = abs(Beta);

   if ( Alpha > 0 )
      return Probs[Alpha-1][Beta-1];

   if ( Alpha < 0 ) // assuming CPT!!!
      return Probs[Beta-1][Alpha-1];

   std::cerr << " BargerPropagator::GetVacuumProb neutrino must be non-zero: " << std::endl;
   return -1.0;

}



void BargerPropagator::propagateLinear( int NuFlavor, double pathlength, double Density )
{

   int    i,j;

   double TransitionMatrix[3][3][2];
   double TransitionProduct[3][3][2];
   double TransitionTemp[3][3][2];
   double RawInputPsi[3][2];
   double OutputPsi[3][2];

   if( ! kSuppressWarnings )
   if(
       ( kAntiMNSMatrix && NuFlavor > 0) ||
       (!kAntiMNSMatrix && NuFlavor < 0)
     )
   {
      std::cout << " Warning BargerPropagator::propagateLinear - " << std::endl;
      std::cout << "     Propagating neutrino flavor and MNS matrix definition differ :" << std::endl;
      std::cout << "     MNS Matrix was defined for : " << ( kAntiMNSMatrix ? " Nubar " : "Nu" )<< std::endl;
      std::cout << "     Propagation is for         : " << ( NuFlavor < 0   ? " Nubar " : "Nu" )<< std::endl;
      std::cout << "     Please check your call to BargerPropagator::SetMNS() " << std::endl;
      std::cout << "     This message can be suppressed with a call to BargerPropagator::SuppressWarnings() " << std::endl;

      exit(-1);
   }

   clear_complex_matrix( TransitionMatrix );
   clear_complex_matrix( TransitionProduct );
   clear_complex_matrix( TransitionTemp );

   ClearProbabilities();


   get_transition_matrix( NuFlavor,
                  Energy	,		// in GeV
                  Density * density_convert,
                  pathlength ,      	// in km
                  TransitionMatrix,	// Output transition matrix
                  0.0
            );

   copy_complex_matrix( TransitionMatrix , TransitionProduct );

   for ( i = 0 ; i < 3 ; i++ )
   {
      for ( j = 0 ; j < 3 ; j++ )
      {       RawInputPsi[j][0] = 0.0; RawInputPsi[j][1] = 0.0;   }

      if( kUseMassEigenstates )
         convert_from_mass_eigenstate( i+1, NuFlavor,  RawInputPsi );
      else
         RawInputPsi[i][0] = 1.0;

      multiply_complex_matvec( TransitionProduct, RawInputPsi, OutputPsi );

      Probability[i][0] += OutputPsi[0][0] * OutputPsi[0][0] + OutputPsi[0][1]*OutputPsi[0][1];
      Probability[i][1] += OutputPsi[1][0] * OutputPsi[1][0] + OutputPsi[1][1]*OutputPsi[1][1];
      Probability[i][2] += OutputPsi[2][0] * OutputPsi[2][0] + OutputPsi[2][1]*OutputPsi[2][1];

   }// end of loop on neutrino types

}


/* 
 * @IN:
 * double *ecen:        Array of energy bins.
 * int ecen_length:     Length of ecen.
 * double *czcen:       Array of other bins.
 * int czcen_length:    Length of czcen.
 * double energy_scale: Energy off scaling due to mis-calibration.
 * double deltam21:     deltam21 value [eV^2].
 * double deltam31:     deltam31 value [eV^2].
 * double deltacp:      deltaCP value to use [rad].
 * double prop_height:  Height in the atmosphere to begin in km.
 * double YeI:          Ye (elec frac) in inner core.
 * double YeO:          Ye (elec frac) in outer core.
 * double YeM:          Ye (elec frac) in mantle.
 * int prop_length:     Length of the output array probList.
 * int evals_length:    Length of the output array evals.
 * int czvals_length:   Length of the output array czvals.
 * double theta12:      theta12 value [rad].
 * double theta13:      theta13 value [rad].
 * double theta23:      theta23 value [rad].
 * 
 * @OUT:
 * double *probList:    The complete probability list over all iterations.
 * double *evals: 
 * double *czvals:
 */
void BargerPropagator::fill_osc_prob_c(double *ecen, int ecen_length, double *czcen, 
        int czcen_length, double energy_scale, double deltam21, double deltam31,
        double deltacp, double prop_height, double YeI, double YeO, double YeM,
        double *probList, int prop_length, double *evals, int evals_length, 
        double *czvals, int czvals_length, double theta12, double theta13, 
        double theta23)
{
    bool kSquared = true;
    double sin2th12Sq = sin(theta12);
    sin2th12Sq = sin2th12Sq*sin2th12Sq;
    double sin2th13Sq = sin(theta13);
    sin2th13Sq = sin2th13Sq*sin2th13Sq;
    double sin2th23Sq = sin(theta23);
    sin2th23Sq = sin2th23Sq*sin2th23Sq;
    
    double *eva= evals;
    double *czva = czvals;
    
    for(int e=0; e<ecen_length; e++)
    {
        // Prepare the arrays which shall be returned.
        memcpy(czva, czcen, czcen_length*sizeof(double));
        czva += czcen_length;
        std::fill_n(eva, czcen_length, ecen[e]);
        eva += czcen_length;
        
        for(int c=0; c<czcen_length; c++)
        {
            double scaled_energy = ecen[e]*energy_scale;
        
            // In BargerPropagator code, it takes the "atmospheric
            // mass difference"-the nearest two mass differences, so
            // that it takes as input deltam31 for IMH and deltam32
            // for NMH
            double mAtm = (deltam31 < 0.0) ? deltam31 : (deltam31 - deltam21);
            
            ////////  FIRST FOR NEUTRINOS ////////
            int kNuBar = 1; // +1 for nu, -1 for nubar
            SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                    deltacp,scaled_energy,kSquared,kNuBar);
            DefinePath(czcen[c], prop_height, YeI, YeO, YeM);
            propagate(kNuBar);
            
            // Copy the probability map to the return array.
            int index = 18*c+e*czcen_length*18;
            std::copy(&Probability[0][0], &Probability[0][0]+9, probList+index);
            ////////  NEXT FOR ANTINEUTRINOS ////////
            kNuBar = -1;
            SetMNS(sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,mAtm,
                   deltacp,scaled_energy,kSquared,kNuBar);
            DefinePath(czcen[c], prop_height, YeI, YeO, YeM);
            propagate(kNuBar);
            // Copy the probability map to the return array.
            std::copy(&Probability[0][0], &Probability[0][0]+9, probList+9+index);
        }
    }
}