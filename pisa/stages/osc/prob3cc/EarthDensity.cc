#include "EarthDensity.h"
#include <iostream>
#include <cstdlib>


EarthDensity::EarthDensity( )
{
    cout << "EarthDensity::EarthDensity Using Default density profile  " << endl;

    _TraverseDistance     = NULL;
    _TraverseRhos         = NULL;
    _TraverseElectronFrac = NULL;

    DetectorDepth = 0.0;

    // radius: [ km ]      density  [ g/cm^3 ]
    _density[ 0 ]       =  13.0 ;
    _density[ 1220.0 ]  =  13.0 ;
    _density[ 3480.0 ]  =  11.3 ;
    _density[ 5701.0 ]  =  5.0 ;
    _density[ 6371.0 ]  =  3.3 ;

    _YeI = 0.4656;
    _YeO = 0.4656;
    _YeM = 0.4957;

    init();
}


EarthDensity::EarthDensity( const char * file, fType _detectorDepth )
{
    _TraverseDistance     = NULL;
    _TraverseRhos         = NULL;
    _TraverseElectronFrac = NULL;

    DetectorDepth         = _detectorDepth;  // [km]

    LoadDensityProfile( file );
}


void EarthDensity::LoadDensityProfile( const char * file )
{
    ifstream PREM_dat;
    fType r_dist;          // radial distance -- map key //
    fType rho;             // density at that distance -- map value //

    DensityFileName = file;

    PREM_dat.open(DensityFileName.c_str());
    if ( ! PREM_dat ) {
        cerr<<"EarthDensity::Load ERROR OPENING " << DensityFileName << endl;
        exit(1);
    }
    //else
    //  cout << "Loading Density profile from: " << DensityFileName << endl;

    while( !PREM_dat.eof( ) ) {
        PREM_dat >> r_dist >> rho ;
        _density[r_dist] = rho;
    }
    PREM_dat.close();

    //cout<<"Profile loaded..."<<endl;

    // must be re-initialized after a profile is loaded
    init();
}


void EarthDensity::init()
{
    Load();
    ComputeMinLengthToLayers();
}


///// Really need to clean this bit up, slow and bulky!
void EarthDensity::SetDensityProfile(fType CosineZ, fType PathLength,
                                     fType ProductionHeight)
/*
  10 May 2014 (TCA): Changed this code so that it calculates the path
  through the atmosphere and if the detectorDepth is greater than
  zero, the path through the outermost layer as well.
*/
{
    int i;
    int MaxLayer;
    fType km2cm = 1.0e5;
    fType TotalEarthLength =  -2.0*CosineZ*RDetector*km2cm; // in [cm]  -YES check - TCA
    fType CrossThis, CrossNext;
    fType default_elec_frac = 0.5;

    map<fType, fType>::iterator _i;

    // TCA: Correctly handle above horizon, through outermost layer...
    if ( CosineZ >= 0 ) {
        // Path through the air:
        fType kappa = DetectorDepth/RDetector;
        fType lambda = CosineZ + sqrt(CosineZ*CosineZ - 1 + (1+kappa)*(1+kappa));
        lambda*=(km2cm*RDetector);
        fType pathThroughAtm = (ProductionHeight*(ProductionHeight + 2.0*DetectorDepth*km2cm +
                                2.0*RDetector*km2cm))/(PathLength + lambda);
        fType pathThroughOuterLayer = PathLength - pathThroughAtm;
        _TraverseRhos[0] = 0.0;
        _TraverseDistance[0] =  pathThroughAtm;
        _TraverseElectronFrac[0] = default_elec_frac;
        Layers = 1;

        if (DetectorDepth > MinDetectorDepth) {
            _TraverseRhos[1] = _Rhos[0];
            _TraverseDistance[1] = pathThroughOuterLayer;
            _TraverseElectronFrac[1] = _YeFrac[_YeFrac.size()-1];
            Layers+=1;
        }

        return;
    }

    // path through air
    _TraverseRhos[0] = 0.0;
    _TraverseDistance[0] = ProductionHeight*(ProductionHeight + DetectorDepth*km2cm +
                                             2.0*RDetector*km2cm)/PathLength;
    _TraverseElectronFrac[0] = default_elec_frac;
    int iTrav = 1;

    // path through the final layer above the detector (if necessary)
    // Note: outer top layer is assumed to be the same as the next layer inward.
    if (DetectorDepth > MinDetectorDepth) {
        _TraverseRhos[1] = _Rhos[0];
        _TraverseDistance[1] = PathLength - TotalEarthLength - _TraverseDistance[0];
        _TraverseElectronFrac[1] = _YeFrac[_YeFrac.size()-1];
        iTrav += 1;
    }

    Layers = 0;
    for ( _i = _CosLimit.begin(); _i != _CosLimit.end() ; _i++ ) {
       if ( CosineZ < _i->second )
           Layers++;
    }

    MaxLayer = Layers;

    // the zeroth layer is the air!
    // and the first layer is the top layer (if detector is not on surface)
    for ( i = 0 ; i< MaxLayer ; i++ ) {
        _TraverseRhos[i+iTrav]      = _Rhos[i];
        _TraverseElectronFrac[i+iTrav] = default_elec_frac;
        for (int iRad = 0; iRad < _YeOuterRadius.size(); iRad++) {
            if (_Radii[i] < (_YeOuterRadius[iRad]*1.001)) {
                _TraverseElectronFrac[i+iTrav] = _YeFrac[iRad];
                break;
            }
        }

        CrossThis = 2.0*sqrt( _Radii[i]*_Radii[i]  - RDetector*RDetector*( 1 -CosineZ*CosineZ ) );
        CrossNext = 2.0*sqrt( _Radii[i+1]*_Radii[i+1]      - RDetector*RDetector*( 1 -CosineZ*CosineZ ) );

        if ( i < MaxLayer-1 )
            _TraverseDistance[i+iTrav]  =  0.5*( CrossThis-CrossNext )*km2cm;
        else
            _TraverseDistance[i+iTrav]  =  CrossThis*km2cm;

        // assumes azimuthal symmetry
        if ( i < MaxLayer ) {
            int index = 2*MaxLayer - i + iTrav - 1;
            _TraverseRhos        [ index ] = _TraverseRhos[i+iTrav-1];
            _TraverseDistance    [ index ] = _TraverseDistance[i+iTrav-1];
            _TraverseElectronFrac[ index ] = _TraverseElectronFrac[i+iTrav-1];
        }
    }

    Layers = 2*MaxLayer + iTrav - 1;

    /* TESTING PURPOSES */
    //cout<<" Layers: "<<Layers<<endl;
    //cout<< " _TraverseRhos, _TraverseDistance, _TraverseElectronFrac: "<<endl;
    //for (int i=0; i<Layers; i++) {
    // cout<<"  "<<_TraverseRhos[i]<<",    "<<_TraverseDistance[i]/km2cm<<",      "<<
    //    _TraverseElectronFrac[i]<<endl;
    //}
}


void EarthDensity::SetElecFrac(fType YeI, fType YeO, fType YeM)
{
    if (_YeOuterRadius.size() != 3) {
        cerr<<"\nERROR: Expects only 3 regions of variable electron fraction! \n"<<
              "  received "<<_YeFrac.size()<<" insead!"<<endl;
        exit(1);
    }

    _YeFrac[0] = YeI;
    _YeFrac[1] = YeO;
    _YeFrac[2] = YeM;
}


// now using Zenith angle to compute minimum conditions...20050620 rvw
void EarthDensity::ComputeMinLengthToLayers()
{
    fType x;

    _CosLimit.clear();

    // first element of _Radii is largest radius!
    for(int i=0; i < (int) _Radii.size() ; i++ ) {
        // Using a cosine threshold instead! //
        x = -1* sqrt( 1 - (_Radii[i] * _Radii[i] / ( RDetector*RDetector)) );
        if ( i  == 0 ) x = 0;
        _CosLimit[ _Radii[i] ] = x;
    }
}


void EarthDensity::Load()
{
    int MaxDepth = 0;

    map<fType, fType>::reverse_iterator _i;
    _i = _density.rbegin();
    REarth = _i->first;

    // TODO...
    // WHYYYYYYYYYYYYYYYYY????????????????
    MinDetectorDepth = 1.0e-3; // <-- Why? // [km] so min is ~ 1 m

    //////////////////////////////////////////////////////////
    // TCA - 08 May, 2014
    // To account for detector depth, we will assume that detector lies
    // in the final layer, and if this is not the case, then raise
    // error, and quit.
    // Otherwise, modify the _density map so that the final layer is the one
    // that is defined by the radius from earth's center to the detector.
    if (DetectorDepth < MinDetectorDepth) {
        DetectorDepth = MinDetectorDepth;
    }
    else {
        std::map<fType,fType>::reverse_iterator rit;

        rit = _density.rbegin();
        fType largest_radius = rit->first;
        fType last_rho = rit->second;
        _density.erase(largest_radius);
        largest_radius -= DetectorDepth;

        // Check if there are any radii greater than this new final layer
        // and if so, quit.
        for (std::map<fType,fType>::iterator it = _density.begin(); it!=_density.end(); ++it) {
            if (it->first > largest_radius) {
                cerr<<"ERROR! detector is placed too deep-no support for multiple layers "
                    <<"above detector"<<endl;
                cerr<<"layer radius: "<<it->first<<" detector radius: "
                    <<largest_radius<<endl;
                exit(1);
            }
        }
        // If not...then we're good to go.
        _density[largest_radius] = last_rho;
    }

    RDetector = REarth - DetectorDepth;
    //cout<<"RDetector: "<<RDetector<<" DetectorDepth: "<<DetectorDepth<<endl;

    if ( _TraverseRhos         != NULL ) delete [] _TraverseRhos;
    if ( _TraverseDistance     != NULL ) delete [] _TraverseDistance;
    if ( _TraverseElectronFrac != NULL ) delete [] _TraverseElectronFrac;

    // Define electron fraction in each layer.
    // The most up-to-date model for the electron fraction (0 - 1) in each layer is:
    // Yearth = 0.4957, Youter_core = 0.4656, Yinner_core = 0.4656),
    // Outer radius [km]: fraction
    _YeOuterRadius.push_back(1121.5); _YeFrac.push_back(0.0);
    _YeOuterRadius.push_back(3480.0); _YeFrac.push_back(0.0);
    _YeOuterRadius.push_back(RDetector); _YeFrac.push_back(0.0);
    SetElecFrac(0.4656, 0.4656, 0.4957);

    // to get the densities in order of decreasing radii
    for( _i = _density.rbegin() ; _i != _density.rend() ; ++_i ) {
        _Rhos.push_back( _i->second );
        _Radii.push_back( _i->first  );
        MaxDepth++;
    }

    // Max number of total layers = 2*(concentric layers below the detector) + 1 (atm)
    int MAXLAYERS = 2*MaxDepth + 1;
    if (DetectorDepth >= MinDetectorDepth) MAXLAYERS += 1;

    _TraverseRhos         = new fType [ MAXLAYERS ];
    _TraverseDistance     = new fType [ MAXLAYERS ];
    _TraverseElectronFrac = new fType [ MAXLAYERS ];

    return;
}


EarthDensity::~EarthDensity( )
{
    if ( _TraverseRhos         != NULL ) delete [] _TraverseRhos;
    if ( _TraverseDistance     != NULL ) delete [] _TraverseDistance;
    if ( _TraverseElectronFrac != NULL ) delete [] _TraverseElectronFrac;
}
