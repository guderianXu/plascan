#pragma once

#include <QDialog>
#include <QJsonObject>

#include <memory>

namespace Ui
{
class AerialTriangulationDialog;
}

class AerialTriangulationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AerialTriangulationDialog(QWidget *parent = nullptr);
    ~AerialTriangulationDialog() override;

    void setImageCount(int count);
    void setReferencePreselectionAvailable(bool available,
                                           int cameraCount = 0,
                                           int imageCount = 0);
    void applySettings(const QJsonObject &settings);
    QJsonObject collectSettings() const;

signals:
    void settingsChanged(const QJsonObject &settings);

private:
    void setupUi();
    void setAdvancedExpanded(bool expanded);
    void emitSettingsChanged();

    std::unique_ptr<Ui::AerialTriangulationDialog> _ui;
    bool _applyingSettings = false;
    bool _referencePreselectionAvailable = false;
};
