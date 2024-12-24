// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // ------------- TODO/A completer ----------
    IntPoint2 actual_point;
    int w = 0;
    Window W;
    while (true)
    {
        int button=anyGetMouse(actual_point,W,w);

        if(button==3){
            // if right click
            break;

        }else {
            if (W==w1){ // window one
                pts1.push_back(actual_point);
                setActiveWindow(w1);
                drawString(actual_point, to_string(pts1.size()), RED);
                fillCircle(actual_point, 3, GREEN);
                cout << " point " << pts1.size() << " with coordinates " << actual_point << " on first window is saved" << endl;
            }
            else{ // window two
                pts2.push_back(actual_point);
                setActiveWindow(w2);
                drawString(actual_point, to_string(pts2.size()), RED);
                fillCircle(actual_point, 3, GREEN);
                cout << " point " << pts2.size() << " with coordinates " << actual_point << " on second window is saved" << endl;
            }
        }
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);
    // ------------- TODO/A completer ----------
    //fill in matrix A and  vector B based on point correspondences
    for(size_t i=0; i<n*2; i+=2) {

        double x_1 = pts1[i/2].x();
        double y_1 = pts1[i/2].y();
        double x_2 = pts2[i/2].x();
        double y_2 = pts2[i/2].y();

        //We fill the i-th row of matrix A and the i-th element of vector B.
        A(i,0) = x_1;A(i,1) = y_1; A(i,2) = 1;A(i,3) = 0; A(i,4) = 0; A(i,5) = 0; A(i,6) = -x_2*x_1; A(i,7) = -x_2*y_1;
        B[i] = x_2;

        //We fill the (i+1)-th row of matrix A and the (i+1)-th element of vector B.
        A(i+1,0) = 0; A(i+1,1) = 0; A(i+1,2) = 0; A(i+1,3) = x_1; A(i+1,4) = y_1; A(i+1,5) = 1; A(i+1,6) = -y_2*x_1; A(i+1,7) = -y_2*y_1;
        B[i+1] = y_2;
    }
    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}

// This function restores the overlap zone
float computeAlpha(int x, int x_overlap_start, int x_overlap_end) {

    if (x < x_overlap_start)
        return 0.0f;  // Completely in image 1
    if (x > x_overlap_end)
        return 1.0f;  // Completely in image 2

    // Linearly interpolate alpha between 0 and 1 in the overlap region
    return (x - x_overlap_start) / float(x_overlap_end - x_overlap_start);
}

// Panorama construction
void panorama(const Image<Color, 2>& I1, const Image<Color, 2>& I2, const Matrix<float>& H) {
    Matrix<float> H_inverse = inverse(H);
    Vector<float> v(3);
    float x0 = 0, y0 = 0, x1 = I2.width(), y1 = I2.height();

    // Project corners of I1 to the panorama space
    v[0] = 0; v[1] = 0; v[2] = 1;
    v = H * v; v /= v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0] = I1.width(); v[1] = 0; v[2] = 1;
    v = H * v; v /= v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0] = I1.width(); v[1] = I1.height(); v[2] = 1;
    v = H * v; v /= v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0] = 0; v[1] = I1.height(); v[2] = 1;
    v = H * v; v /= v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    // Display dimensions of the panorama
    cout << "x0 x1 y0 y1 = " << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1 << endl;

    // Create new panorama image with computed bounds
    Image<Color> I(static_cast<int>(x1 - x0), static_cast<int>(y1 - y0));
    setActiveWindow(openWindow(I.width(), I.height()));
    I.fill(WHITE);

    // Iterate over the new panorama image
    for (int y = static_cast<int>(y0); y < static_cast<int>(y1); y++) {
        for (int x = static_cast<int>(x0); x < static_cast<int>(x1); x++) {
            bool belongsToI1 = false;
            bool belongsToI2 = false;

            // Prepare point for current pixel
            Vector<float> current_point(3);
            current_point[0] = x;
            current_point[1] = y;
            current_point[2] = 1;

            // Check if the current pixel is within bounds of I2
            if (x >= 0 && x < I2.width() && y >= 0 && y < I2.height()) {
                I(x - static_cast<int>(x0), y - static_cast<int>(y0)) = I2.interpolate(x, y);
                belongsToI2 = true;
            }

            // Transform the current point using the inverse homography matrix
            Vector<float> current_point_inver = H_inverse * current_point;
            current_point_inver /= current_point_inver[2];  // Normalize

            // Check if the transformed point is within bounds of I1
            if (current_point_inver[0] >= 0 && current_point_inver[0] < I1.width() &&
                current_point_inver[1] >= 0 && current_point_inver[1] < I1.height()) {
                I(x - static_cast<int>(x0), y - static_cast<int>(y0)) = I1.interpolate(current_point_inver[0], current_point_inver[1]);
                belongsToI1 = true;
            }

            // If point belongs to both I1 and I2, blend them
            if (belongsToI1 == true && belongsToI2 == true){
                // Check overlap between I1 and I2 and blend if necessary
                float overlapStart = max(0.0f, x0);
                float overlapEnd = min(I1.width(), int(I2.width() + x0));
                float alpha = computeAlpha(x, overlapStart, overlapEnd);  // Compute blending factor based on position
                int panorama_x = x - static_cast<int>(x0);
                int panorama_y = y - static_cast<int>(y0);

                // Blend the red, green, and blue channels
                I(panorama_x, panorama_y).r() = (1-alpha) * I1.interpolate(current_point_inver[0], current_point_inver[1]).r() + alpha * I2.interpolate(x, y).r();
                I(panorama_x, panorama_y).g() = (1-alpha) * I1.interpolate(current_point_inver[0], current_point_inver[1]).g() + alpha * I2.interpolate(x, y).g();
                I(panorama_x, panorama_y).b() = (1-alpha) * I1.interpolate(current_point_inver[0], current_point_inver[1]).b() + alpha * I2.interpolate(x, y).b();
            }
        }
    }

    // Display the resulting panorama
    display(I, 0, 0);
}




// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;


    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
