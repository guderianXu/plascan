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
class FramePinholeCamera;
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
    bool buildCameraFromImageMeta(const QJsonObject &imgObj, xjw::FramePinholeCamera *cam, QString *errorMsg) const;
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

    ProjectManager *_projectManager{};

    QComboBox *_image1Combo{};
    QComboBox *_image2Combo{};
    QComboBox *_pickModeCombo{};
    QLabel *_hintLabel{};
    QPushButton *_deleteSelectedBtn{};
    QPushButton *_clearManualBtn{};
    QPushButton *_runBtn{};
    QTableWidget *_pairTable{};
    QTableWidget *_resultTable{};
    QTabWidget *_tabWidget{};
    DualImageViewer *_viewer{};

    QVector<QPointF> _manualPts1;
    QVector<QPointF> _manualPts2;
    QVector<QPointF> _currentPts1;
    QVector<QPointF> _currentPts2;
    QVector<xjw::Intersection::Result> _currentResults;
    bool _currentPairsEditable{false};
    // 右键配对临时状态：如果 firstSide==0 表示已在左侧选了点，1 表示右侧
    int _pendingFirstSide{-1};
    QPointF _pendingFirstPoint{};
    int _currentHighlighted{-1};
    int _resultSortCol{-1};
    Qt::SortOrder _resultSortOrder{Qt::DescendingOrder};
};
