#include "detection/AprilTagDetector.h"
#include "detection/MarkerDetectorFactory.h"

#include <apriltag/apriltag.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/tag36h11.h>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <stdexcept>

namespace xjw::control_points
{
    namespace
    {
        cv::Mat tagImage()
        {
            const std::unique_ptr<apriltag_family_t, decltype(&tag36h11_destroy)> family(tag36h11_create(),
                                                                                         tag36h11_destroy);
            const std::unique_ptr<image_u8_t, decltype(&image_u8_destroy)> native(apriltag_to_image(family.get(), 7),
                                                                                  image_u8_destroy);
            cv::Mat scaled;
            cv::resize(cv::Mat(native->height, native->width, CV_8UC1, native->buf, native->stride),
                       scaled,
                       cv::Size(144, 144),
                       0.0,
                       0.0,
                       cv::INTER_NEAREST);
            cv::Mat image(192, 192, CV_8UC1, cv::Scalar(255));
            scaled.copyTo(image(cv::Rect(24, 24, 144, 144)));
            return image;
        }

        const QVector<MarkerTargetFamily> families = {MarkerTargetFamily::AprilTag36h11,
                                                      MarkerTargetFamily::NonCodedCircle,
                                                      MarkerTargetFamily::NonCodedFourQuadrant};
    } // namespace

    TEST(MarkerDetectorInputTest, AprilTagHandlesRotatedNonContiguousViewsWithoutChangingInput)
    {
        cv::Mat image = tagImage();
        MarkerDetectionOptions options;
        options.threadCount = 1;
        for (int rotation = 0; rotation < 4; ++rotation)
        {
            cv::Mat backing(200, 220, CV_8UC1, cv::Scalar(17));
            cv::Mat view = backing(cv::Rect(3, 4, 192, 192));
            image.copyTo(view);
            ASSERT_FALSE(view.isContinuous());
            const cv::Mat before = backing.clone();
            const AprilTagDetector detector(AprilTagFamily::Tag36h11);
            const auto strided = detector.detect(view, {}, options);
            const auto contiguous = detector.detect(view.clone(), {}, options);
            ASSERT_EQ(strided.size(), 1);
            ASSERT_EQ(contiguous.size(), 1);
            EXPECT_EQ(strided.front().targetId, 7);
            EXPECT_EQ(strided.front().center, contiguous.front().center);
            EXPECT_EQ(strided.front().corners, contiguous.front().corners);
            EXPECT_DOUBLE_EQ(strided.front().decisionMargin, contiguous.front().decisionMargin);
            EXPECT_EQ(cv::countNonZero(backing != before), 0);
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
        }
    }

    TEST(MarkerDetectorInputTest, NonzeroStridedMaskExcludesAprilTagCenter)
    {
        cv::Mat backing(200, 220, CV_8UC1, cv::Scalar(0));
        cv::Mat mask = backing(cv::Rect(3, 4, 192, 192));
        // Cover both sides of the half-pixel center so floating-point fit rounding
        // cannot move the exclusion onto a neighboring unmasked pixel.
        mask(cv::Rect(94, 94, 4, 4)).setTo(1);
        MarkerDetectionOptions options;
        options.threadCount = 1;
        EXPECT_TRUE(AprilTagDetector(AprilTagFamily::Tag36h11).detect(tagImage(), mask, options).isEmpty());
    }

    TEST(MarkerDetectorInputTest, RejectsUnsupportedImageAndMaskTypesForEveryImplementation)
    {
        const cv::Mat image(40, 50, CV_8UC1, cv::Scalar(255));
        for (const MarkerTargetFamily family : families)
        {
            const auto detector = MarkerDetectorFactory::create(family);
            EXPECT_THROW(detector->detect(cv::Mat(40, 50, CV_8UC3), {}, {}), std::invalid_argument);
            EXPECT_THROW(detector->detect(cv::Mat(40, 50, CV_16UC1), {}, {}), std::invalid_argument);
            EXPECT_THROW(detector->detect(image, cv::Mat(40, 50, CV_8UC3), {}), std::invalid_argument);
            EXPECT_THROW(detector->detect(image, cv::Mat(39, 50, CV_8UC1), {}), std::invalid_argument);
            const int dimensions[] = {2, 40, 50};
            EXPECT_THROW(detector->detect(cv::Mat(3, dimensions, CV_8UC1), {}, {}), std::invalid_argument);
            EXPECT_THROW(detector->detect(image, cv::Mat(3, dimensions, CV_8UC1), {}), std::invalid_argument);
        }
    }

    TEST(MarkerDetectorInputTest, EmptyAndCancelledInputsReturnNoObservations)
    {
        std::atomic_bool cancelled = true;
        MarkerDetectionOptions options;
        options.cancelRequested = &cancelled;
        for (const MarkerTargetFamily family : families)
        {
            const auto detector = MarkerDetectorFactory::create(family);
            EXPECT_TRUE(detector->detect({}, {}, {}).isEmpty());
            EXPECT_TRUE(detector->detect(cv::Mat(40, 50, CV_8UC1), {}, options).isEmpty());
        }
    }
} // namespace xjw::control_points
