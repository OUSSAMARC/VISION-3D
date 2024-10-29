// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Calcule le nombre optimal d'itérations pour l'algorithme RANSAC.
int getOptimalIterations(int currentIterNb, int inlierCount,
                         int sampleSize, int matchCount) {
    // Vérification : éviter les divisions par zéro ou log négatif.
    if (matchCount == 0 || inlierCount == 0 || sampleSize <= 0) {
        return currentIterNb;
    }

    // Calcul du ratio des inliers et vérification des bornes.
    float inlierRatio = static_cast<float>(inlierCount) / matchCount;
    if (inlierRatio <= 0.0f || inlierRatio >= 1.0f) {
        return currentIterNb;
    }
    // Calcul du nouveau nombre d'itérations.
    int nextIterNb = static_cast<int>(
        std::log(BETA) / std::log(1.0f - std::pow(inlierRatio, sampleSize)));

    // Limiter à l'itération courante si nécessaire.
    return std::min(nextIterNb, currentIterNb);
}


// Estimation of the fundamental matrix F with matrix A always being square.
FMatrix<float, 3, 3> estimateFundamentalMatrixSVD(const std::vector<Match>& matches, vector<int>& inds) {
    // Check: At least 8 correspondences are required.
    if (matches.size() < 8) {
        std::cerr << "Error: At least 8 correspondences are required." << std::endl;
        return FMatrix<float, 3, 3>(0.0f);
    }
    // Define a scaling matrix for normalization.
    FMatrix<float, 3, 3> normMatrix(0.0f);
    normMatrix(0, 0) = 0.001f;
    normMatrix(1, 1) = 0.001f;
    normMatrix(2, 2) = 1.0f;

    // Initial size of the correspondences.
    int m = inds.size();
    int n = std::max(m, 9);

    // Define a square matrix A (n x n).
    Matrix<float> A(n, n); // SVD is easier with a square matrix
    A.fill(0.0f);  // Initialize all values to 0.

    // Fill A with the linear constraints according to the available correspondences.
    for (int i = 0; i < m; ++i) {
        // Create normalized homogeneous points.
        FVector<float, 3> p1 = {matches[inds[i]].x1, matches[inds[i]].y1, 1.0f};
        FVector<float, 3> p2 = {matches[inds[i]].x2, matches[inds[i]].y2, 1.0f};

        // Apply normalization.
        p1 = normMatrix * p1;
        p2 = normMatrix * p2;

        // Fill the corresponding row in matrix A.
        A(i, 0) = p1[0] * p2[0];
        A(i, 1) = p1[0] * p2[1];
        A(i, 2) = p1[0];
        A(i, 3) = p1[1] * p2[0];
        A(i, 4) = p1[1] * p2[1];
        A(i, 5) = p1[1];
        A(i, 6) = p2[0];
        A(i, 7) = p2[1];
        A(i, 8) = 1.0f;
    }

    Vector<float> D;
    Matrix<float> U, V;
    svd(A, U, D, V);
    Vector<float> f = V.getRow(8);
    FMatrix<float, 3, 3> F;
    F(0, 0) = f[0]; F(0, 1) = f[1]; F(0, 2) = f[2];
    F(1, 0) = f[3]; F(1, 1) = f[4]; F(1, 2) = f[5];
    F(2, 0) = f[6]; F(2, 1) = f[7]; F(2, 2) = f[8];

    FVector<float, 3> B;
    FMatrix<float, 3, 3> E, C;
    svd(F, E, B, C);
    B[2] = 0;

    F = transpose(normMatrix) * E * Diagonal(B) * C * normMatrix;

    return F;
}



vector<int> Inliers(FMatrix<float, 3, 3>& F, vector<Match>& matches, float distMax) {
    vector<int> inliers;
    float norm;

    // Loop through all matches to find inliers.
    for (int i = 0; i < matches.size(); i++) {
        Match currentMatch = matches[i];
        FloatPoint3 point1, point2;

        // Create homogeneous points from the match.
        point1[0] = currentMatch.x1;
        point1[1] = currentMatch.y1;
        point1[2] = 1;
        point2[0] = currentMatch.x2;
        point2[1] = currentMatch.y2;
        point2[2] = 1;

        // Compute the epipolar line for point1.
        FVector<float, 3> line;
        line = transpose(F) * point1;

        // Calculate the geometric distance from point2 to the epipolar line.
        float dist = abs(line[0] * point2[0] + line[1] * point2[1] + line[2]);
        norm = sqrt(line[0] * line[0] + line[1] * line[1]);
        dist /= norm;

        // If the distance is within the allowed threshold, add the match as an inlier.
        if (dist <= distMax) {
            inliers.push_back(i);
        }
    }

    return inliers;
}


// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter `matches` is filtered to keep only inliers as output.
FMatrix<float, 3, 3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter = 100000; // Adjusted dynamically
    FMatrix<float, 3, 3> bestF;
    vector<int> bestInliers;

    // --------------- TODO ------------
    // DO NOT FORGET NORMALIZATION OF POINTS
    int nInliersMax = 0;

    // Iterate Niter times to find the best fundamental matrix.
    for (int iter = 0; iter < Niter; iter++) {
        // Select a random subset of 8 unique indices.
        vector<int> inds;
        vector<vector<int>> Vectors;  // Store previously used index sets.
        for (int i = 0; i < 8; ++i) {
            int id;
            id = rand() % matches.size();
            // Add the unique index to the vector.
            inds.push_back(id);
        }
        // If the selected subset already exists in `Vectors`, generate a new one.
        if (std::find(Vectors.begin(), Vectors.end(), inds) != Vectors.end()) {
            --iter;  // Redo the iteration with a new random subset.
            continue;
        }

        // Store the current subset to avoid reusing it.
        Vectors.push_back(inds);
        // Estimate the fundamental matrix F using the selected matches.
        FMatrix<float, 3, 3> F = estimateFundamentalMatrixSVD(matches, inds);

        // Find inliers for the estimated matrix F.
        vector<int> Inliershere = Inliers(F, matches, distMax);

        // If the current set of inliers is the largest so far, update the best solution.
        if (Inliershere.size() > nInliersMax) {
            nInliersMax = Inliershere.size();
            cout << "Ok" << endl; // Debug print

            bestF = F;
            bestInliers = Inliershere;

            // Adjust the number of iterations based on the inlier ratio.
            Niter = getOptimalIterations(Niter, matches.size(), 8, Inliershere.size());
        }
    }

    // Refine F using the best set of inliers.
    bestF = estimateFundamentalMatrixSVD(matches, bestInliers);

    // Update `matches` to keep only the inliers.
    vector<Match> all = matches;
    matches.clear();
    for (size_t i = 0; i < bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    return bestF;
}


// Expects clicks in one image and shows the corresponding epipolar line in the other image.
// Stops on right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float, 3, 3>& F) {
    while (true) {
        int x, y;
        // Stop the loop if the user right-clicks (event code 3).
        if (getMouse(x, y) == 3)
            break;

        // Create a homogeneous point (x, y, 1) from the clicked point.
        DoublePoint3 point1 = {x, y, 1};

        // Generate a random color for the epipolar line.
        Color couleur(rand() % 256, rand() % 256, rand() % 256);

        FVector<float, 3> line;
        bool Imagedroite = (x >= I1.width());

        // Compute the epipolar line depending on the click location.
        if (Imagedroite) {
            point1[0] -= I1.width();  // Adjust x-coordinate relative to the right image.
            line = F * point1;        // Epipolar line in the left image.
        } else {
            line = transpose(F) * point1;  // Epipolar line in the right image.
        }

        // Compute the endpoints of the epipolar line.
        float x1 = 0;
        float y1 = -line[2] / line[1];
        float x2 = I1.width();
        float y2 = (-line[0] * I1.width() - line[2]) / line[1];

        // Adjust line position if the click was on the left image.
        if (!Imagedroite) {
            x1 += I1.width();
            x2 += I1.width();
        }

        // Display the clicked point and the corresponding epipolar line.
        drawCircle(x, y, 2, couleur);
        drawLine(x1, y1, x2, y2, couleur);
    }
}


int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();

    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}

