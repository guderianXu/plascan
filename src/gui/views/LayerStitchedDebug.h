#pragma once

#include <QString>

class QGraphicsPixmapItem;
class QGraphicsScene;

namespace xjw::gui::views
{

void recordStitchedImagePairDebug(QGraphicsScene *scene,
                                  const QString &plascanPath,
                                  const QString &pathA,
                                  const QString &pathB,
                                  const QGraphicsPixmapItem *itemA,
                                  const QGraphicsPixmapItem *itemB,
                                  int gap);

} // namespace xjw::gui::views
