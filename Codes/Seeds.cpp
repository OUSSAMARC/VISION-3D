// Imagine++ project
// Project:  Seeds
// Author:   Pascal Monasse
// Student: TODO

#include <Imagine/Images.h>
#include <queue>
#include <string>
#include <iostream>
using namespace Imagine;
using namespace std;

// Default data
const char *DEF_im1=srcPath("face0.png"), *DEF_im2=srcPath("face1.png");
static int dmin=-30, dmax=-7; // Min and max disparities

/// Min NCC for a seed
static const float nccSeed=0.95f;

/// Radius of patch for correlation
static const int win=(9-1)/2;
/// To avoid division by 0 for constant patch
static const float EPS=0.1f;

/// A seed
struct Seed {
    Seed(int x0, int y0, int d0, float ncc0)
        : x(x0), y(y0), d(d0), ncc(ncc0) {}
    int x,y, d;
    float ncc;
};

/// Order by NCC
bool operator<(const Seed& s1, const Seed& s2) {
    return (s1.ncc<s2.ncc);
}

/// 4-neighbors
static const int dx[]={+1,  0, -1,  0};
static const int dy[]={ 0, -1,  0, +1};

/// Display disparity map
static Image<Color> displayDisp(const Image<int>& disp, Window W, int subW) {
    Image<Color> im(disp.width(), disp.height());
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++) {
            if(disp(i,j)<dmin || disp(i,j)>dmax)
                im(i,j) = CYAN;
            else {
                int g = 255*(disp(i,j)-dmin)/(dmax-dmin);
                im(i,j)= Color(g,g,g);
            }
        }
    setActiveWindow(W,subW);
    display(im);
    showWindow(W,subW);
    return im;
}

/// Show 3D window
static void show3D(const Image<Color>& im, const Image<int>& disp) {
#ifdef IMAGINE_OPENGL // Imagine++ must have been built with OpenGL support...
    // Intrinsic parameters given by Middlebury website
    const float f=3740;
    const float d0=-200; // Doll images cropped by this amount
    const float zoom=2; // Half-size images, should double measured disparity
    const float B=0.160; // Baseline in m
    FMatrix<float,3,3> K(0.0f);
    K(0,0)=-f/zoom; K(0,2)=disp.width()/2;
    K(1,1)= f/zoom; K(1,2)=disp.height()/2;
    K(2,2)=1.0f;
    K = inverse(K);
    K /= K(2,2);
    std::vector<FloatPoint3> pts;
    std::vector<Color> col;
    for(int j=0; j<disp.height(); j++)
        for(int i=0; i<disp.width(); i++)
            if(dmin<=disp(i,j) && disp(i,j)<=dmax) {
                float z = B*f/(zoom*disp(i,j)+d0);
                FloatPoint3 pt((float)i,(float)j,1.0f);
                pts.push_back(K*pt*z);
                col.push_back(im(i,j));
            }
    Mesh mesh(&pts[0], pts.size(), 0,0,0,0,VERTEX_COLOR);
    mesh.setColors(VERTEX, &col[0]);
    Window W = openWindow3D(512,512,"3D");
    setActiveWindow(W);
    showMesh(mesh);
#else
    std::cout << "No 3D: Imagine++ not built with OpenGL support" << std::endl;
#endif
}

/// Correlation between patches centered on (i1,j1) and (i2,j2). The values
/// m1 or m2 are subtracted from each pixel value.
static float correl(const Image<byte>& im1, int i1, int j1, float m1,
                    const Image<byte>& im2, int i2, int j2, float m2) {
    float dist=0.0f;
    float numerator = 0.0f;
    float denominator1 = 0.0f;
    float denominator2 = 0.0f;

    for (int r = -win; r < win; ++r) {
        for (int c = -win; c < win; ++c) {

            float IL_val = im1(i1 + r, j1 + c) - m1;
            float IR_val = im2(i2 + r, j2 + c) - m2;

            numerator += IL_val * IR_val;
            denominator1 += pow(IL_val ,2);
            denominator2 += pow(IR_val ,2);

        }
    }

    // Calculate the denominator
    float denominator = sqrt(denominator1 * denominator2  + EPS);

    // Return the normalized cross-correlation value
    dist = numerator / denominator;
    return dist;
}


/// Sum of pixel values in patch centered on (i,j).
static float sum(const Image<byte>& im, int i, int j) {
    float sum=0.0f;
    // ------------- TODO -------------
    for(int x = i-win; x < i+win+1; x++){
        for(int y = j-win; y < j+win+1; y++)
        {
            sum += (float)im(x, y);
        }
    }
    return sum;

}

/// Centered correlation of patches of size 2*win+1.
static float ccorrel(const Image<byte>& im1,int i1,int j1,
                     const Image<byte>& im2,int i2,int j2) {
    float m1 = sum(im1,i1,j1);
    float m2 = sum(im2,i2,j2);
    int w = 2*win+1;
    return correl(im1,i1,j1,m1/(w*w), im2,i2,j2,m2/(w*w));
}

/// Compute disparity map from im1 to im2, but only at points where NCC is
/// above nccSeed. Set to true the seeds and put them in Q.
static void find_seeds(Image<byte> im1, Image<byte> im2,
                       float nccSeed,
                       Image<int>& disp, Image<bool>& seeds,
                       std::priority_queue<Seed>& Q) {
    disp.fill(dmin-1);
    seeds.fill(false);
    while(!Q.empty())
        Q.pop();

    const int maxy = std::min(im1.height(), im2.height());
    const int refreshStep = (maxy - 2*win) * 5 / 100;
    for(int y = win; y + win < maxy; y++) {
        if((y - win - 1) / refreshStep != (y - win) / refreshStep)
            std::cout << "Seeds: " << 5 * (y - win) / refreshStep <<"%\r"<<std::flush;
        for(int x = win; x + win < im1.width(); x++) {
            // ------------- TODO -------------
            // Hint: just ignore windows that are not fully in image
            float bestNCC = -1.0f;
            int bestD = dmin;

            for (int d = dmin; d <= dmax; d++) {
                if (x + d - win >= 0 && x + d + win < im2.width()) {
                    float ncc = ccorrel(im1, x, y, im2, x + d, y);
                    if (ncc > bestNCC) {
                        bestNCC = ncc;
                        bestD = d;

                    }
                }
            }

            if (bestNCC >= nccSeed) {
                disp(x, y) = bestD;
                seeds(x, y) = true;
                Q.push(Seed(x, y, bestD, bestNCC));
            }
        }
    }
    std::cout << std::endl;
}


/// Propagate seeds
static void propagate(Image<byte> im1, Image<byte> im2,
                      Image<int>& disp, Image<bool>& seeds,
                      std::priority_queue<Seed>& Q)
{
    while (!Q.empty()) {
        Seed s = Q.top();
        Q.pop();
        for (int i = 0; i < 4; i++) {
            int x = s.x + dx[i];
            int y = s.y + dy[i];
            if (x >= win && y >= win && x + win < im2.width() && y + win < im2.height() && !seeds(x, y)) {
                // ------------- TODO -------------
                float bestD = s.d;
                float bestCorrelation = -1.0f;

                // Iterate over a small range of disparities
                for (int W = -1; W <= 2; W++) {
                    if( x+W+s.d>win )
                    {
                        float Ncc = ccorrel(im1, x, y, im2, x + W + s.d, y);
                        if (Ncc > bestCorrelation) {
                            bestCorrelation = Ncc;
                            bestD = W + s.d;
                            // Limit bestD to be within [dmin, dmax]
                            if (bestD < dmin) {
                                bestD = dmin;
                            } else if (bestD > dmax) {
                                bestD = dmax;
                            }
                        }
                    }
                }

                seeds(x, y) = true;
                disp(x, y) = bestD;
                Q.push(Seed(x, y, bestD, bestCorrelation));
            }
        }
    }
}


int main()
{
    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, srcPath("face0.png")) ||
        ! load(I2, srcPath("i")) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    std::string names[5]={"image 1","image 2","dense","seeds","propagation"};
    Window W = openComplexWindow(I1.width(), I1.height(), "Seeds propagation",
                                 5, names);
    setActiveWindow(W,0);
    display(I1,0,0);
    setActiveWindow(W,1);
    display(I2,0,0);

    Image<int> disp(I1.width(), I1.height());
    Image<bool> seeds(I1.width(), I1.height());
    std::priority_queue<Seed> Q;

    // Dense disparity
    cout<<"Processing dense disparity..."<<endl;
    find_seeds(I1, I2, -1.0f, disp, seeds, Q);
    displayDisp(disp,W,2);

    // Only seeds
    cout<<"Finding Seeds..."<<endl;
    find_seeds(I1, I2, nccSeed, disp, seeds, Q);
    displayDisp(disp,W,3);

    // Propagation of seeds
    cout<<"Seeds propagation..."<<endl;
    propagate(I1, I2, disp, seeds, Q);
    displayDisp(disp,W,4);

    cout<<"Done !"<<endl;
    cout<<"Shift + Click on 3D Model to rotate. ( click to END )"<<endl;
    // Show 3D (use shift click to animate)
    show3D(I1,disp);

    endGraphics();
    return 0;
}
