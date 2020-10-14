#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <math.h>      
#include <limits>       
#include <unordered_map>
#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"
#include <map>
using namespace std;

std::vector< Vec3 > positions;
std::vector< Vec3 > normals;
std::vector< Vec3 > positionsToProject;
std::vector< Vec3 > normalsToProject;
std::vector< Vec3 > positionsProjected;
std::vector< Vec3 > normalsProjected;
float pointSize = 2.f;
bool drawInputPointset = true;
bool drawPointsetToProject = true;
bool drawPointsetProjected = true;

//START OF GLOABAL VARIABLES//
int N=64;
        float max_x=std::numeric_limits<float>::min() ;
        float max_y=std::numeric_limits<float>::min() ; 
        float max_z=std::numeric_limits<float>::min() ; 
        float min_x=std::numeric_limits<float>::max() ; 
        float min_y=std::numeric_limits<float>::max() ; 
        float min_z=std::numeric_limits<float>::max() ; 

        std::vector<float>gridF(N*N*N);
        float pas_x,pas_y,pas_z;
        
        std::map<std::vector<float>,int> centers; 
        std::vector<Vec3>mesh_positions; 
        std::vector<unsigned int> mesh_triangles; 
        int id=0;


///Multiresolution Variables; 
bool MR=false; // to be set to true to test the multiresolution function  

        std::map<std::vector<float>,int> MR_centers; 
        std::vector<Vec3>MR_mesh_positions; 
        std::vector<unsigned int> MR_mesh_triangles; 
        int MR_id=0;

//END OF GLOABAL VARIABLES//

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;




// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}
void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}
void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}


bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}










// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glDisable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (1.f, 1.f, 1.f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}


//---------------------------------------------------------------------//
// The following function can be used to render a triangle mesh
// It takes as input:
// - a set of 3d points
// - a set of point indices: 3 consecutive indices make a triangle.
//   for example, if i_triangles = { 0 , 4 , 1 , 3 , 2 , 5 , ... },
//   then the first two triangles will be cclear omposed of
//     {i_positions[0], i_positions[4] , i_positions[1] } and
//     {i_positions[3], i_positions[2] , i_positions[5] }
void drawTriangleMesh( std::vector< Vec3 > const & i_positions , std::vector< unsigned int > const & i_triangles ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_triangles.size() / 3 ; ++tIt) {
        Vec3 p0 = i_positions[i_triangles[3*tIt]];
        Vec3 p1 = i_positions[i_triangles[3*tIt+1]];
        Vec3 p2 = i_positions[i_triangles[3*tIt+2]];
        Vec3 n = Vec3::cross(p1-p0 , p2-p0);
        n.normalize();
        glNormal3f( n[0] , n[1] , n[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();
}

// The following function can be used to render a pointset
void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}



void draw () {
   glPointSize(pointSize); // for example...

    glColor3f(0.8,0.8,1);
    if( drawInputPointset )
        drawPointSet(positions , normals);

    glColor3f(1,0.5,0.5);
    if( drawPointsetToProject )
        drawPointSet(positionsToProject , normalsToProject);

    glColor3f(0.5,0.8,0.5);
    if( drawPointsetProjected )
        drawPointSet(positionsProjected , normalsProjected);
    if(MR){ // if tessting multi resolution 
        drawTriangleMesh(MR_mesh_positions,MR_mesh_triangles);
    }else{
        drawTriangleMesh(mesh_positions,mesh_triangles);

    }
}

void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    case 'a':
        pointSize /= 1.1;
        break;
    case 'z':
        pointSize *= 1.1;
        break;
    case '1':
        drawInputPointset = !drawInputPointset;
        break;
    case '2':
        drawPointsetToProject = !drawPointsetToProject;
        break;
    case '3':
        drawPointsetProjected = !drawPointsetProjected;
        break;

    default:
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}







float Get_kernal(int kernel_type , float r, float h , float w=1.0 ){
      switch (kernel_type) {
    case 0:
        return exp(-r*r/(h*h)); 
        break;
    case 1:
        return (pow((1-r/h),4)*(1+4*r/h)); 
        break;
    case 2:
        return (pow((r/h),-w) );
        break;

}
}



void HPSS( Vec3 inputPoint, // the point you need to project
           Vec3 & outputPoint , Vec3 & outputNormal , // the projection (and the normal at the projection)
           std::vector< Vec3 > const & inputPositions , std::vector< Vec3 > const & inputNormals , // the input pointset on which you want to project
           BasicANNkdTree const & inputPointsetKdTree , // the input pointset kdtree: it can help you find the points that are nearest to a query point
           int kernel_type , float radius, // the parameters for the MLS surface (a kernel can be i) a Gaussian, ii) Wendland, iii) Singular
           unsigned int numberOfIterations , // the number of iterations of the MLS algorithm
           unsigned int knn = 20 ) { // the number of nearest neighbors you will consider when computing the MLS projection
            
            for (int iter=0; iter<numberOfIterations; iter++){
                ANNidxArray id_nearest_neighbors = new ANNidx[knn];
                ANNdistArray square_distances_to_neighbors = new ANNdist[knn];
                inputPointsetKdTree.knearest(inputPoint, knn, id_nearest_neighbors, square_distances_to_neighbors);
                float sum_weight=0.0; 
                Vec3 sum_weighted_neighbors=Vec3(0.0,0.0,0.0); 
                Vec3 sum_weighted_normals=Vec3(0.0,0.0,0.0); 
                for(int i =0 ; i< knn ; i++)
                {
                    Vec3 neighbor = inputPositions[id_nearest_neighbors[i]];
                    Vec3 neighborNormal=inputNormals[id_nearest_neighbors[i]];
                    Vec3 ProjOnTang=inputPoint-(Vec3::dot((inputPoint-neighbor),neighborNormal)*neighborNormal);
                    float weight=Get_kernal(1,sqrt(square_distances_to_neighbors[i]),radius);
                    sum_weighted_neighbors+=weight*ProjOnTang; 
                    sum_weight+=weight;
                    sum_weighted_normals+=weight*neighborNormal; 
                }
                Vec3 c =sum_weighted_neighbors/sum_weight; 
                sum_weighted_normals.normalize();
                Vec3 n= sum_weighted_normals;
                outputPoint=c; 
                outputNormal=n;
                delete[] id_nearest_neighbors;
                delete[] square_distances_to_neighbors;
                inputPoint=outputPoint;
            }
}
 
void APSS( Vec3 inputPoint, // the point you need to project
           Vec3 & outputPoint , Vec3 & outputNormal , // the projection (and the normal at the projection)
           std::vector< Vec3 > const & inputPositions , std::vector< Vec3 > const & inputNormals , // the input pointset on which you want to project
           BasicANNkdTree const & inputPointsetKdTree , // the input pointset kdtree: it can help you find the points that are nearest to a query point
           int kernel_type , float radius, // the parameters for the MLS surface (a kernel can be i) a Gaussian, ii) Wendland, iii) Singular
           unsigned int numberOfIterations , // the number of iterations of the MLS algorithm
           unsigned int knn = 20 ) { // the number of nearest neighbors you will consider when computing the MLS projection

            
        outputPoint = inputPoint;
        for (int iter=0; iter<numberOfIterations; iter++){
            ANNidxArray id_nearest_neighbors = new ANNidx[knn];
            ANNdistArray square_distances_to_neighbors = new ANNdist[knn];
            inputPointsetKdTree.knearest(outputPoint, knn, id_nearest_neighbors, square_distances_to_neighbors);
            Vec3 u123=Vec3(0,0,0);
            float u4=0.0,u0=0.0,sum_1=0.0,sum_4=0.0;
            Vec3 sum_2=Vec3(0,0,0),sum_3=Vec3(0,0,0);
            float sum_weight = 0.0;
            for (unsigned int i=0;i<knn;i++){
                Vec3 neighbor = inputPositions[id_nearest_neighbors[i]];
                Vec3 neighborNormal=inputNormals[id_nearest_neighbors[i]];            
      
                 float w = Get_kernal(kernel_type,sqrt(square_distances_to_neighbors[i]),radius);
                sum_1 = sum_1 +(w*Vec3::dot(neighbor,neighborNormal));
                sum_2 = sum_2 +(w*neighbor);
                sum_3 = sum_3 +(w*neighborNormal);
                sum_weight+=w;
                sum_4 = sum_4 + (w*Vec3::dot(neighbor,neighbor));
            }
            u4 = 0.5*((sum_1-(Vec3::dot(sum_2,sum_3)/sum_weight))/( sum_4-(Vec3::dot(sum_2,sum_2)/sum_weight)));
            u123 = (sum_3-(2*u4*sum_2))/sum_weight;
            u0 = -(Vec3::dot(sum_2,u123)+u4*sum_4)/sum_weight;
            outputNormal = u123 + 2*u4*outputPoint; 
            outputNormal.normalize();
            Vec3 diff=(u0+Vec3::dot(u123,outputPoint)+u4*Vec3::dot(outputPoint,outputPoint))*outputNormal;
            outputPoint = outputPoint-diff;
            delete[] id_nearest_neighbors;
            delete[] square_distances_to_neighbors;
    }
}

// Begin of Functions Needed for Dual contouring function 
Vec3 getPos( float i , float j , float k ){
    /*
    return the postion of the vertex at positon (i,j,k)
    */
    
    float x,y,z; 
    x=min_x+(max_x-min_x)*i/(N-1);
    y=min_y+(max_y-min_y)*j/(N-1);
    z=min_z+(max_z-min_z)*k/(N-1);
    return Vec3(x,y,z);
}

bool check(  int i , int j , int k) { 
    /*
    Funciton to check if the indexs ( i , j , k ) are inside of the grid.
    */
    return (i>=0&&i<N&&j>=0&&j<N&&j>=0&&j<N&&k>=0&&k<N); 
}

bool opposed(  int i , int j , int k, int ii , int jj , int kk){
    /*
    Function to check if value of function F at the vertex of position (i,j,k) 
    is opposed to the value of function F at the vertex of position (ii,jj,kk)
    aka: F(Vertex(i,j,k))*F(vertex(ii,jj,kk)) < 0
    */
     return ((gridF[i+j*N+k*N*N]*gridF[ii+jj*N+kk*N*N])<0);
}

vector<float> apply(float x,float y ,float z){
    /*
        Input: 
            x,y,z: Floats - positions in the grid.
        Output: 
            v: vector of float - position  vector of the vertex at position (x,y,z)
        Function: 
            check if the vertex V at position x,y,z if mapped to an id in the Centers hashmap, 
            if not assign new id to this new vertex and add it to the mesh_position vector.
    */

     vector<float> v {x,y,z}; 
     if(!centers[v]){
        centers[v]=id++;
        mesh_positions.push_back(Vec3(x,y,z)); 
     }
    return v;  
}
int  MR_apply(Vec3 vv){
        /*
        Input: 
            vv: Vec3 - postion vector of a vertex 
        Output: 
            MR_centers[v] : int -  return the id of that vertex in the mapping hashmap(vertex to ID )
        Function: 
            check if the vertex V at position vv if mapped to an id in the MR_Centers hashmap, 
            if not assign new id to this new vertex and add it to the MR_mesh_position vector.
    */
     vector<float> v {vv[0],vv[1],vv[2]}; 
     if(!MR_centers[v]){
        MR_centers[v]=MR_id++;
        MR_mesh_positions.push_back(vv); 
     }
      return  MR_centers[v] ; 
 }
 
 void append_triangle(   std::vector<float> p1,   std::vector<float> p2,   std::vector<float> p3,   std::vector<float> p4, bool up )
 {
    /*
        Input: 
            p1,p2,p3,p4: Vector of floats - postion vector of a vertex 
            up: bool - indicate the direction of the normal for the 
                        surface generated by the two sets of point:
                        Triangle 1 (p1,p2,p3) and Triangle2(p1,p3,p4)
        Function: 
            add the vertexs p1,p2,p3 and p4 with a certain order depending 
            of the direction of the normal of the surface into the mesh_traingles 
            vector to create two triangles with the same normal as the surface of the pointset. 
    */ 
     if(up){
     mesh_triangles.push_back(centers[p1]);
     mesh_triangles.push_back(centers[p2]);
     mesh_triangles.push_back(centers[p3]);
     mesh_triangles.push_back(centers[p1]);
     mesh_triangles.push_back(centers[p3]);
     mesh_triangles.push_back(centers[p4]);

     }else 
    {
      mesh_triangles.push_back(centers[p1]);
     mesh_triangles.push_back(centers[p3]);
     mesh_triangles.push_back(centers[p2]);
     mesh_triangles.push_back(centers[p1]);
     mesh_triangles.push_back(centers[p4]);
     mesh_triangles.push_back(centers[p3]);
    }

}
bool check_normal_dir(int i,int j,int k)
{
    /*
        return the direction of the normal to the surface at vertex (i,j,k)
    */
    return  gridF[i+j*N+k*N*N]<0;
}
// End of Functions Needed for Dual contouring function 


void DC(std::vector< Vec3 > const & inputPositions , std::vector< Vec3 > const & inputNormals , // the input pointset on which you want to project
           BasicANNkdTree const & inputPointsetKdTree , // the input pointset kdtree: it can help you find the points that are nearest to a query point
           int kernel_type , float radius, // the parameters for the MLS surface (a kernel can be i) a Gaussian, ii) Wendland, iii) Singular
           unsigned int numberOfIterations , // the number of iterations of the MLS algorithm
           unsigned int knn = 20 ) { // the number of nearest neighbors you will consider when computing the MLS projection
    /*
        Dual Contouring Function 
    */
    // find the coordinates of the bounding box
    for(int i = 0 ; i < inputPositions.size()  ;i++   ){
        float x= inputPositions[i][0];
        float y= inputPositions[i][1];
        float z= inputPositions[i][2];
        max_x=max(max_x,x);
        max_y=max(max_y,y);
        max_z=max(max_z,z);
        min_x=min(min_x,x);
        min_y=min(min_y,y);
        min_z=min(min_z,z);
        }
    //add padding to the bounding box ( we add one cell along each axis , half cell size  on each side)
        //calculate the padding value 
        float x_pad=(max_x-min_x)/(N-1)/2;
        float y_pad=(max_y-min_y)/(N-1)/2;
        float z_pad=(max_z-min_z)/(N-1)/2;
        //add the padding to the bounding box 
        max_x+=x_pad;
        max_y+=y_pad;
        max_z+=z_pad;
        min_x-=x_pad;
        min_y-=y_pad;
        min_z-=z_pad;
        
        //calculat the distance between the center of the cell(cube) and  one edge of the cell( cube ) on each axis  
        pas_x=(max_x-min_x)/(N-1)/2;
        pas_y=(max_y-min_y)/(N-1)/2;
        pas_z=(max_z-min_z)/(N-1)/2;

        //generate the grid (N*N*N) and calculat the value of F at each Vertex of the grid. 
       for( int i = 0; i < N; i++ )
       { 
            for( int j= 0; j < N; j++ )
            { 
                for( int k = 0; k < N; k++ )
                {   Vec3 oP, oN;
                    Vec3 Vertex_pos=getPos(i,j,k);
                    HPSS( Vertex_pos , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
                    gridF[i+N*j+N*N*k]=Vec3::dot(Vertex_pos-oP,oN);

                }
            }
        }
        /*check each vertex and it's neighbors, if they have opposed sign values of F 
        we create surface arround that edge created by the two vetexs of opposed sign F values
        the surface created by the triangles have the same normal direction as the surface of the pointset  
        */
        for( int i = 0; i < N; i++ )
        { 
            for( int j =0 ; j < N; j++ )
            { 
                for( int k = 0 ; k < N; k++ )
                {   Vec3 pos=getPos(i,j,k);
                    bool up =check_normal_dir(i,j,k);
                    if(check(i+1,j,k)&& opposed(i+1,j,k,i,j,k)){

                                std::vector<float> p1 =apply(pos[0]+pas_x,pos[1]+pas_y,pos[2]+pas_z);
                                std::vector<float> p2 =apply(pos[0]+pas_x,pos[1]-pas_y,pos[2]+pas_z);
                                std::vector<float> p3 =apply(pos[0]+pas_x,pos[1]-pas_y,pos[2]-pas_z);
                                std::vector<float> p4 =apply(pos[0]+pas_x,pos[1]+pas_y,pos[2]-pas_z);
                                append_triangle(p1,p2,p3,p4,up);
                    } 
                    if(check(i,j+1,k)&& opposed(i,j+1,k,i,j,k)){                     
                                std::vector<float> p1 =apply(pos[0]+pas_x,pos[1]+pas_y,pos[2]+pas_z);
                                std::vector<float> p2 =apply(pos[0]+pas_x,pos[1]+pas_y,pos[2]-pas_z);
                                std::vector<float> p3 =apply(pos[0]-pas_x,pos[1]+pas_y,pos[2]-pas_z);
                                std::vector<float> p4 =apply(pos[0]-pas_x,pos[1]+pas_y,pos[2]+pas_z);
                                append_triangle(p1,p2,p3,p4,up) ;
                    } 
                    if(check(i,j,k+1)&& opposed(i,j,k+1,i,j,k)){            
                                std::vector<float> p1 =apply(pos[0]+pas_x,pos[1]+pas_y,pos[2]+pas_z);
                                std::vector<float> p2 =apply(pos[0]-pas_x,pos[1]+pas_y,pos[2]+pas_z);
                                std::vector<float> p3 =apply(pos[0]-pas_x,pos[1]-pas_y,pos[2]+pas_z);
                                std::vector<float> p4 =apply(pos[0]+pas_x,pos[1]-pas_y,pos[2]+pas_z);
                                append_triangle(p1,p2,p3,p4,up);
                    }
                }
            }
        }

}

void multiresolution(std::vector< Vec3 > const & inputPositions , std::vector< Vec3 > const & inputNormals , // the input pointset on which you want to project
           BasicANNkdTree const & inputPointsetKdTree , // the input pointset kdtree: it can help you find the points that are nearest to a query point
           int kernel_type , float radius, // the parameters for the MLS surface (a kernel can be i) a Gaussian, ii) Wendland, iii) Singular
           unsigned int numberOfIterations , // the number of iterations of the MLS algorithm
           unsigned int knn = 20 ) { // the number of nearest neighbors you will consider when computing the MLS projection
    /*
    Multiresolution over Dual contouring  with low resolution 
    */
    
    // we need for applying Multiresolution  the extranction of 
    //the mesh triangles of a low resolution using Dual contouring
    
    //So we clear the data structures used to store the data of the mesh triangles
    mesh_triangles.clear() ; 
    mesh_positions.clear() ; 
    // call the dial contouring function to generate our  low-res mesh triangles
    DC(  inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn );
    
    // loop over the mesh_triangles and divide each triangle into 4 triangles 
    //then project all the vertexs on the pointset the add the 4 triangles to the new MR_mesh_triangles
    for ( int i = 0 ; i< mesh_triangles.size() ; i+=3){
         Vec3 p1=mesh_positions[mesh_triangles[i]];
         Vec3 p2=mesh_positions[mesh_triangles[i+1]];
         Vec3 p3=mesh_positions[mesh_triangles[i+2]];
         Vec3 p12=(p1+p2)/2; 
         Vec3 p23=(p2+p3)/2; 
         Vec3 p13=(p1+p3)/2; 
         Vec3 oP, oN;
         HPSS( p1 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p1=oP;
         HPSS( p2 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p2=oP;
         HPSS( p3 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p3=oP;
        HPSS( p12 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p12=oP;
         HPSS( p23 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p23=oP;
         HPSS( p13 , oP , oN , inputPositions , inputNormals , inputPointsetKdTree , kernel_type, radius, numberOfIterations , knn ); 
         p13=oP;
        
        MR_mesh_triangles.push_back(MR_apply(p1));
        MR_mesh_triangles.push_back(MR_apply(p12));
        MR_mesh_triangles.push_back(MR_apply(p13));

        MR_mesh_triangles.push_back(MR_apply(p12));
        MR_mesh_triangles.push_back(MR_apply(p2));
        MR_mesh_triangles.push_back(MR_apply(p13));

        MR_mesh_triangles.push_back(MR_apply(p2));
        MR_mesh_triangles.push_back(MR_apply(p23));
        MR_mesh_triangles.push_back(MR_apply(p13));

        MR_mesh_triangles.push_back(MR_apply(p23));
        MR_mesh_triangles.push_back(MR_apply(p3));
        MR_mesh_triangles.push_back(MR_apply(p13));
        }
}


int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("3D Rendering");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);


    {
        // Load a first pointset, and build a kd-tree:
        loadPN("pointsets/igea.pn" , positions , normals);

        BasicANNkdTree kdtree;
        kdtree.build(positions);
        double alpha=0.15;
        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        positionsToProject.resize( 20000 );
        normalsToProject.resize(positionsToProject.size());
        for( unsigned int pIt = 0 ; pIt < positionsToProject.size() ; ++pIt ) {
            positionsToProject[pIt] = Vec3(
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX)
                        );
            positionsToProject[pIt].normalize();
            positionsToProject[pIt] = 0.6 * positionsToProject[pIt];
        }

        // IF YOU WANT TO TAKE AS POINTSET TO PROJECT THE INPUT POINTSET ITSELF: USEFUL FOR POINTSET FILTERING
        if(true) {
            positionsToProject = positions;
            normalsToProject = normals;
            bool noise = 0; //SET TO 1 TO ADD NOISE TO THE DATA 
            if(noise){
                for( unsigned int pIt = 0 ; pIt < positionsToProject.size() ; ++pIt ) {
                    double  noise_ratio= alpha* (2.0*((double)rand() / RAND_MAX)-1);
                    positionsToProject[pIt] += (noise_ratio*normals[pIt]);
                }
            }            
        }

        // INITIALIZE THE PROJECTED POINTSET (USEFUL MAINLY FOR MEMORY ALLOCATION)
        positionsProjected.resize(positionsToProject.size());
        normalsProjected.resize(positionsToProject.size());
        mesh_triangles.clear() ; 
        mesh_positions.clear() ; 
        
        // PROJECT USING MLS (HPSS, and later APSS):
           /* 
            for( unsigned int pIt = 0 ; pIt < positionsToProject.size() ; ++pIt ) {
                Vec3 oP, oN;

                //HPSS( positionsToProject[pIt] , oP , oN , positions , normals , kdtree , 0, 100, 10 , 20 );
                APSS( positionsToProject[pIt] , oP , oN , positions , normals , kdtree , 0.1, 100, 10 , 20 ); 

                positionsProjected[pIt] = oP;
                normalsProjected[pIt] = oN;
            }
            */
        // To Change the Resolution , alter the value of N in the begning of the code.
        //To Apply dual Contouring Only 
            //MR=false ; 
            //DC(positions , normals , kdtree , 0, 0.1, 10 , 20 );
        //To Apply multiresolution over dual Contouring with low-Res   
            
            MR=true;
            multiresolution(  positions , normals , kdtree , 0, 0.1, 10 , 20 );
            


    }
    glutMainLoop ();
    return EXIT_SUCCESS;
}


