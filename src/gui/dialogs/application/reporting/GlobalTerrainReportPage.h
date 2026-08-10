#pragma once

#include <QJsonObject>
#include <QWidget>

/**
 * @brief Displays the core-generated small-body global DEM/DOM report.
 *
 * The page is intentionally read-only. It renders metadata from the supplied
 * JSON object and loads the existing preview PNG without rebuilding products.
 */
class GlobalTerrainReportPage : public QWidget
{
public:
    explicit GlobalTerrainReportPage(const QJsonObject &report, QWidget *parent = nullptr);

private:
    void buildUi(const QJsonObject &report);
};
