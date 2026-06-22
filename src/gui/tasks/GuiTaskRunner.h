#pragma once

#include <QMetaObject>
#include <QPointer>
#include <QtConcurrent/QtConcurrent>

#include <memory>
#include <type_traits>
#include <utility>

namespace xjw::gui::tasks
{

template <typename Owner, typename Callback>
void postGuarded(Owner *owner, Callback &&callback)
{
    QPointer<Owner> self(owner);
    if (!self)
    {
        return;
    }

    auto callbackPtr = std::make_shared<std::decay_t<Callback>>(std::forward<Callback>(callback));
    QMetaObject::invokeMethod(self.data(),
                              [self, callbackPtr]() mutable
                              {
                                  if (!self)
                                  {
                                      return;
                                  }
                                  (*callbackPtr)(self.data());
                              },
                              Qt::QueuedConnection);
}

template <typename Owner, typename Work, typename Finished>
void runGuarded(Owner *owner, Work &&work, Finished &&finished)
{
    QPointer<Owner> self(owner);
    if (!self)
    {
        return;
    }

    using WorkFn = std::decay_t<Work>;
    using FinishedFn = std::decay_t<Finished>;
    using Result = std::invoke_result_t<WorkFn &>;

    auto workPtr = std::make_shared<WorkFn>(std::forward<Work>(work));
    auto finishedPtr = std::make_shared<FinishedFn>(std::forward<Finished>(finished));

    (void)QtConcurrent::run([self, workPtr, finishedPtr]() mutable
    {
        if (!self)
        {
            return;
        }

        if constexpr (std::is_void_v<Result>)
        {
            (*workPtr)();
            if (!self)
            {
                return;
            }
            postGuarded(self.data(),
                        [finishedPtr](Owner *owner) mutable
                        {
                            (*finishedPtr)(owner);
                        });
        }
        else
        {
            auto resultPtr = std::make_shared<Result>((*workPtr)());
            if (!self)
            {
                return;
            }
            postGuarded(self.data(),
                        [finishedPtr, resultPtr](Owner *owner) mutable
                        {
                            (*finishedPtr)(owner, std::move(*resultPtr));
                        });
        }
    });
}

} // namespace xjw::gui::tasks
