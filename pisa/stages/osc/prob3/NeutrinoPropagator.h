#ifndef __NeutrinoPropagator__
#define __NeutrinoPropagator__

#include <iostream>

using namespace std;
class NeutrinoPropagator
{

   public:
     NeutrinoPropagator(){};

     virtual ~NeutrinoPropagator(){};
     virtual void propagate( int  )
               { cerr<< " Warning NeutrinoPropagator::propagate( int ) undefined " << endl; }
     virtual void propagateLinear( int, double, double )
               { cerr<< " Warning NeutrinoPropagator::propagateLinear( int ) undefined " << endl; }
     virtual void DefinePath( double , double , bool kSetProfile = true  )
               { cerr<< " Warning NeutrinoPropagator::DefinePath(double, double, bool) undefined " << endl; }

     virtual void SetMNS( double , double , double , double , double , double , double, bool, int kNuType = 1 )
               { cerr<< " Warning NeutrinoPropagator::SetMNS(double,..., int ) undefined " << endl; }

     virtual double GetProb( int , int )
               { cerr<< " Warning NeutrinoPropagator::GetProb(int, int) undefined " << endl; return 0.0; }
               
     virtual double GetVacuumProb( int , int , double , double )
               { cerr<< " Warning NeutrinoPropagator::GetVacuumProb(int, int, double , double ) undefined " << endl; return 0.0; }

     virtual void SetPotential( double [][3][2] )
               { cerr<< " Warning NeutrinoPropagator::SetPotential(double [][3][2]) undefined " << endl; }

     virtual double GetPathLength() 
               { cerr<< " Warning NeutrinoPropagator::GetPathLength( ) undefined " << endl; return 0.0; }

     int WhoAmI(){ return IAm; }

     int IAm;

};


#endif
