#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там
// пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
//    throw std::runtime_error("not implemented yet 4.2");
    int rows = 2 * count, cols = 3;
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd b(rows);

    for (int i = 0; i < count; i++) {
        int i2 = i * 2;
        A(i2, 0) = ms[i][0] * Ps[i](2, 0) - ms[i][2] * Ps[i](0, 0);
        A(i2 + 1, 0) = ms[i][1] * Ps[i](2, 0) - ms[i][2] * Ps[i](1, 0);

        A(i2, 1) = ms[i][0] * Ps[i](2, 1) - ms[i][2] * Ps[i](0, 1);
        A(i2 + 1, 1) = ms[i][1] * Ps[i](2, 1) - ms[i][2] * Ps[i](1, 1);

        A(i2, 2) = ms[i][0] * Ps[i](2, 2) - ms[i][2] * Ps[i](0, 2);
        A(i2 + 1, 2) = ms[i][1] * Ps[i](2, 2) - ms[i][2] * Ps[i](1, 2);

        b(i2) = - ms[i][0] * Ps[i](2, 3) + ms[i][2] * Ps[i](0, 3);
        b(i2 + 1) = - ms[i][1] * Ps[i](2, 3) + ms[i][2] * Ps[i](1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd D(cols, rows);
    for (int i = 0; i < rows; i++) {
        D(0, i) = 0;
        D(1, i) = 0;
        D(2, i) = 0;
    }
    D(0, 0) = 1 / svda.singularValues()[0];
    D(1, 1) = 1 / svda.singularValues()[1];
    D(2, 2) = 1 / svda.singularValues()[2];

    auto matrix = svda.matrixV() * D * svda.matrixU().transpose() * b;
    return cv::Vec4d{matrix[0], matrix[1], matrix[2], 1};
}
