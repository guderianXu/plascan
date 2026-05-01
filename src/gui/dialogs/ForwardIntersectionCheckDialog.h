#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QPointF>
#include <QVector>

#include "Intersection.h"

class ProjectManager;
class QComboBox;
class QLabel;
class QPushButton;
class QTableWidget;
class QTabWidget;
class DualImageViewer;

namespace xjw {
class Camera;
}

class ForwardIntersectionCheckDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ForwardIntersectionCheckDialog(ProjectManager *projectManager, QWidget *parent = nullptr);
    ~ForwardIntersectionCheckDialog() override;

private slots:
    void onImageSelectionChanged();
    void onClearManualPoints();
    void onRunCheck();
    void onViewerLeftRightClicked(const QPointF &scenePos);
    void onViewerRightRightClicked(const QPointF &scenePos);
    void onPairTableClicked(int row, int column);
    void onResultTableClicked(int row, int column);
    void onDeleteSelectedPairs();
    void onViewerPointClicked(int index);
    void onResultTableHeaderClicked(int col);

private:
    void setupUi();
    void loadImagesWithCamera();
    bool collectAutoPointPairs(QVector<QPointF> *pts1, QVector<QPointF> *pts2, QString *sourceInfo);
    bool buildCameraFromImageMeta(const QJsonObject &imgObj, xjw::Camera *cam, QString *errorMsg) const;
    QJsonObject findImageMetaByPath(const QString &imagePath) const;
    void refreshViewer(bool reloadImages);
    void refreshPairTable();
    void fillResultTable(const QVector<xjw::Intersection::Result> &results);
    void fillResultTableOrdered(const QVector<int> &order);
    void applyPendingPointHint();
    void clearAllSelections();
    QJsonObject buildBatchResultJson(const QVector<QPointF> &pts1,
                                     const QVector<QPointF> &pts2,
                                     const QVector<xjw::Intersection::Result> &results,
                                     const QString &mode,
                                     const QString &autoSource) const;
    QString selectedImage1() const;
    QString selectedImage2() const;

    ProjectManager *m_projectManager{};

    QComboBox *m_image1Combo{};
    QComboBox *m_image2Combo{};
    QComboBox *m_pickModeCombo{};
    QLabel *m_hintLabel{};
    QPushButton *m_deleteSelectedBtn{};
    QPushButton *m_clearManualBtn{};
    QPushButton *m_runBtn{};
    QTableWidget *m_pairTable{};
    QTableWidget *m_resultTable{};
    QTabWidget *m_tabWidget{};
    DualImageViewer *m_viewer{};

    QVector<QPointF> m_manualPts1;
    QVector<QPointF> m_manualPts2;
    QVector<QPointF> m_currentPts1;
    QVector<QPointF> m_currentPts2;
    QVector<xjw::Intersection::Result> m_currentResults;
    bool m_currentPairsEditable{false};
    // 右键配对临时状态：如果 firstSide==0 表示已在左侧选了点，1 表示右侧
    int m_pendingFirstSide{-1};
    QPointF m_pendingFirstPoint{};
    int m_currentHighlighted{-1};
    int m_resultSortCol{-1};
    Qt::SortOrder m_resultSortOrder{Qt::DescendingOrder};
};
