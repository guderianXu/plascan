#pragma once

#include "placamera/errors.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace placamera
{

    /** Result for recoverable non-numerical operations. */
    template <typename T> class Result
    {
    public:
        static Result success(T value)
        {
            Result result;
            result._value = std::move(value);
            return result;
        }

        static Result failure(CameraError error)
        {
            Result result;
            result._error = std::move(error);
            return result;
        }

        static Result failure(CameraErrorCode code,
                              std::string message,
                              std::string source = {},
                              std::optional<std::size_t> line = std::nullopt)
        {
            return failure({code, std::move(message), std::move(source), line});
        }

        bool ok() const noexcept
        {
            return _value.has_value();
        }

        explicit operator bool() const noexcept
        {
            return ok();
        }

        const T& value() const&
        {
            return _value.value();
        }

        T& value() &
        {
            return _value.value();
        }

        T&& value() &&
        {
            return std::move(_value).value();
        }

        const T& operator*() const&
        {
            return value();
        }

        T& operator*() &
        {
            return value();
        }

        const T* operator->() const
        {
            return &value();
        }

        T* operator->()
        {
            return &value();
        }

        T takeValue()
        {
            return std::move(_value).value();
        }

        const CameraError& error() const noexcept
        {
            return _error;
        }

        CameraErrorCode errorCode() const noexcept
        {
            return _error.code;
        }

        const std::string& message() const noexcept
        {
            return _error.message;
        }

    private:
        Result() = default;

        std::optional<T> _value;
        CameraError _error;
    };

    template <> class Result<void>
    {
    public:
        static Result success()
        {
            Result result;
            result._ok = true;
            return result;
        }

        static Result failure(CameraError error)
        {
            Result result;
            result._error = std::move(error);
            return result;
        }

        static Result failure(CameraErrorCode code,
                              std::string message,
                              std::string source = {},
                              std::optional<std::size_t> line = std::nullopt)
        {
            return failure({code, std::move(message), std::move(source), line});
        }

        bool ok() const noexcept
        {
            return _ok;
        }

        explicit operator bool() const noexcept
        {
            return ok();
        }

        const CameraError& error() const noexcept
        {
            return _error;
        }

        CameraErrorCode errorCode() const noexcept
        {
            return _error.code;
        }

        const std::string& message() const noexcept
        {
            return _error.message;
        }

    private:
        Result() = default;

        bool _ok = false;
        CameraError _error;
    };

    struct EvaluationWarning
    {
        std::string code;
        std::string message;
    };

    template <typename T> class EvaluationResult
    {
    public:
        static EvaluationResult success(T value, double achievedPrecisionPixels = 0.0)
        {
            EvaluationResult result;
            result._value = std::move(value);
            result._achievedPrecisionPixels = achievedPrecisionPixels;
            return result;
        }

        static EvaluationResult failure(CameraErrorCode code, std::string message)
        {
            return failure({code, std::move(message), {}, std::nullopt});
        }

        static EvaluationResult failure(CameraError error)
        {
            EvaluationResult result;
            result._error = std::move(error);
            return result;
        }

        bool ok() const noexcept
        {
            return _value.has_value();
        }

        explicit operator bool() const noexcept
        {
            return ok();
        }

        const T& value() const
        {
            return _value.value();
        }

        T& value()
        {
            return _value.value();
        }

        CameraErrorCode errorCode() const noexcept
        {
            return _error.code;
        }

        const std::string& message() const noexcept
        {
            return _error.message;
        }

        const CameraError& error() const noexcept
        {
            return _error;
        }

        double achievedPrecisionPixels() const noexcept
        {
            return _achievedPrecisionPixels;
        }

        const std::vector<EvaluationWarning>& warnings() const noexcept
        {
            return _warnings;
        }

        void addWarning(EvaluationWarning warning)
        {
            _warnings.push_back(std::move(warning));
        }

    private:
        EvaluationResult() = default;

        std::optional<T> _value;
        CameraError _error;
        double _achievedPrecisionPixels = 0.0;
        std::vector<EvaluationWarning> _warnings;
    };

} // namespace placamera
