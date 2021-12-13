#include "ConfusionMatrix.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cassert>

std::string ConfusionMatrix::COL_SEP("\t");
std::string ConfusionMatrix::ROW_BEGIN("\t");
std::string ConfusionMatrix::ROW_END("");

ConfusionMatrix::ConfusionMatrix() {}

ConfusionMatrix::ConfusionMatrix(const int m) {
    _matrix.resize(m);
    for (int i = 0; i < m; i++) {
        _matrix[i].resize(m, 0);
    }
}

ConfusionMatrix::~ConfusionMatrix() {}

int ConfusionMatrix::numRows() const {
    if (_matrix.empty())
        return 0;
    return (int)_matrix.size();
}

int ConfusionMatrix::numCols() const {
    if (_matrix.empty())
        return 0;
    return (int)_matrix[0].size();
}

void ConfusionMatrix::resize(const int m) {
    _matrix.resize(m);
    for (int i = 0; i < m; i++) {
        _matrix[i].resize(m, 0);
    }
}

void ConfusionMatrix::clear() {
    for (size_t i = 0; i < _matrix.size(); ++i) {
        std::fill(_matrix[i].begin(), _matrix[i].end(), 0);
    }
}

double ConfusionMatrix::rowSum(int n) const {
    double v = 0.0;
    for (size_t i = 0; i < _matrix[n].size(); i++) {
        v += (double)_matrix[n][i];
    }
    return v;
}

double ConfusionMatrix::colSum(int m) const {
    double v = 0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        v += (double)_matrix[i][m];
    }
    return v;
}

double ConfusionMatrix::diagSum() const {
    double v = 0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (i >= _matrix[i].size())
            break;
        v += (double)_matrix[i][i];
    }
    return v;
}

double ConfusionMatrix::totalSum() const {
    double v = 0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        for (size_t j = 0; j < _matrix[i].size(); j++) {
            v += (double)_matrix[i][j];
        }
    }
    return v;
}

void ConfusionMatrix::accumulate(const int actual, const int predicted)
{
    assert(actual >= 0);
    assert(predicted >= 0);

    _matrix[actual][predicted] += 1;

}

void ConfusionMatrix::accumulate(const ConfusionMatrix& confusion) {
    assert(confusion._matrix.size() == _matrix.size());

    if (_matrix.empty()) return;

    assert(confusion._matrix[0].size() == _matrix[0].size());

    for (size_t row = 0; row < _matrix.size(); ++row) {
        for (size_t col = 0; col < _matrix[row].size(); ++col) {
            _matrix[row][col] += confusion._matrix[row][col]; 
        }
    }
}

void ConfusionMatrix::printCounts(const char* header) const {
    if (header == NULL) {
        std::cout << "--- confusion matrix: (actual, predicted) ---" << std::endl;
    }
    else {
        std::cout << header << std::endl;
    }
    std::stringstream ss;

    for (size_t i = 0; i < _matrix.size(); i++) {
        for (size_t j = 0; j < _matrix[i].size(); j++) {
            if (j > 0) {
                ss << COL_SEP << " | " << COL_SEP;
            }
            ss << _matrix[i][j];
        }
        ss << "\n";
    }

    std::cout << ss.str() << std::endl;
}

void ConfusionMatrix::printRowNormalized(const char* header) const {
    if (header == NULL) {
        std::cout << "--- confusion matrix: (actual, predicted) ---" << std::endl;
    }
    else {
        std::cout << header << std::endl;
    }
    for (size_t i = 0; i < _matrix.size(); i++) {
        double total = rowSum(i);
        std::cout << ROW_BEGIN;
        for (size_t j = 0; j < _matrix[i].size(); j++) {
            if (j > 0) {
                std::cout << COL_SEP;
            }
            std::cout << ((double)_matrix[i][j] / total);
        }
        std::cout << ROW_END;
    }
}

void ConfusionMatrix::printColNormalized(const char* header) const {
    std::vector<double> totals;
    for (size_t i = 0; i < _matrix[0].size(); i++) {
        totals.push_back(colSum(i));
    }

    if (header == NULL) {
        std::cout << "--- confusion matrix: (actual, predicted) ---" << std::endl;
    }
    else {
        std::cout << header << std::endl;
    }
    for (size_t i = 0; i < _matrix.size(); i++) {
        std::cout << ROW_BEGIN << std::endl;
        for (size_t j = 0; j < _matrix[i].size(); j++) {
            if (j > 0) {
                std::cout << COL_SEP;
            }
            std::cout << ((double)_matrix[i][j] / totals[j]);
        }
        std::cout << ROW_END;
    }
}

void ConfusionMatrix::printNormalized(const char* header) const {
    double total = totalSum();

    if (header == NULL) {
        std::cout << "--- confusion matrix: (actual, predicted) ---";
    }
    else {
        std::cout << header;
    }
    for (size_t i = 0; i < _matrix.size(); i++) {
        std::cout << ROW_BEGIN;
        for (size_t j = 0; j < _matrix[i].size(); j++) {
            if (j > 0) std::cout << COL_SEP;
            std::cout << ((double)_matrix[i][j] / total);
        }
        std::cout << ROW_END;
    }
}

void ConfusionMatrix::printPrecisionRecall(const char* header) const {
    if (header == NULL) {
        std::cout << "--- class-specific recall/precision ---";
    }
    else {
        std::cout << header;
    }

    // recall
    std::cout << ROW_BEGIN;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (i > 0) {
            std::cout << COL_SEP;
        }
        double r = (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)rowSum(i) : 0.0;
        std::cout << r;
    }
    std::cout << ROW_END;

    // precision
    std::cout << ROW_BEGIN;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (i > 0) {
            std::cout << COL_SEP;
        }
        double p = (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)colSum(i) : 1.0;
        std::cout << p;
    }
    std::cout << ROW_END;
}

void ConfusionMatrix::printF1Score(const char* header) const {
    if (header == NULL) {
        std::cout << "--- class-specific F1 score ---";
    }
    else {
        std::cout << header;
    }

    std::cout << ROW_BEGIN;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (i > 0) {
            std::cout << COL_SEP;
        }
        // recall
        double r = (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)rowSum(i) : 0.0;
        // precision
        double p = (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)colSum(i) : 1.0;
        std::cout << ((2.0 * p * r) / (p + r));
    }
    std::cout << ROW_END;
}


void ConfusionMatrix::printJaccard(const char* header) const {
    if (header == NULL) {
        std::cout << "--- class-specific Jaccard coefficient ---";
    }
    else {
        std::cout << header;
    }

    std::cout << ROW_BEGIN;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (i > 0) {
            std::cout << COL_SEP;
        }
        double p = (_matrix[i].size() > i) ? (double)_matrix[i][i] /
            (double)(rowSum(i) + colSum(i) - _matrix[i][i]) : 0.0;
        std::cout << p;
    }
    std::cout << ROW_END;
}

double ConfusionMatrix::accuracy() const {
    double total_sum = totalSum();
    double diag_sum = diagSum();

    if (total_sum == 0) {
        return 0;
    }
    else {
        return diag_sum / total_sum;
    }
}

double ConfusionMatrix::avgPrecision() const {
    double totalPrecision = 0.0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        totalPrecision += (_matrix[i].size() > i) ?
            (double)_matrix[i][i] / (double)colSum(i) : 1.0;
    }

    return totalPrecision /= (double)_matrix.size();
}

double ConfusionMatrix::avgRecall(const bool strict) const {
    double totalRecall = 0.0;
    int numClasses = 0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (_matrix[i].size() > i) {
            const double classSize = (double)rowSum(i);
            if (classSize > 0.0) {
                totalRecall += (double)_matrix[i][i] / classSize;
                numClasses += 1;
            }
        }
    }

    if (strict && numClasses != (int)_matrix.size()) {
        std::cout << "not all classes represented in avgRecall()";
    }

    if (numClasses == 0) {
        return 0;
    }
    else {
        return totalRecall / (double)numClasses;
    }
}

double ConfusionMatrix::avgJaccard() const {
    double totalJaccard = 0.0;
    for (size_t i = 0; i < _matrix.size(); i++) {
        if (_matrix[i].size() <= i) continue;
        const double intersectionSize = (double)_matrix[i][i];
        const double unionSize = (double)(rowSum(i) + colSum(i) - _matrix[i][i]);
        if (intersectionSize == unionSize) {
            // avoid divide by zero
            totalJaccard += 1.0;
        }
        else {
            totalJaccard += intersectionSize / unionSize;
        }
    }

    return totalJaccard / (double)_matrix.size();
}

double ConfusionMatrix::precision(int n) const {
    assert(_matrix.size() > (size_t)n);
    return (_matrix[n].size() > (size_t)n) ?
        (double)_matrix[n][n] / (double)colSum(n) : 1.0;
}

double ConfusionMatrix::recall(int n) const {
    assert(_matrix.size() > (size_t)n);
    return (_matrix[n].size() > (size_t)n) ?
        (double)_matrix[n][n] / (double)rowSum(n) : 0.0;
}

double ConfusionMatrix::jaccard(int n) const {
    assert((_matrix.size() > (size_t)n) && (_matrix[n].size() > (size_t)n));
    const double intersectionSize = (double)_matrix[n][n];
    const double unionSize = (double)(rowSum(n) + colSum(n) - _matrix[n][n]);
    return (intersectionSize == unionSize) ? 1.0 :
        intersectionSize / unionSize;
}

const unsigned long& ConfusionMatrix::operator()(int i, int j) const {
    return _matrix[i][j];
}

unsigned long& ConfusionMatrix::operator()(int i, int j) {
    return _matrix[i][j];
}