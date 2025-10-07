class QnnMockSSRController {
    public:
        enum class Timing {
            TensorCreateGraphTensor,
            GraphAddNode,
            GraphFinalize,
            GraphExecute
        };

        static QnnMockSSRController& Instance() {
            static QnnMockSSRController instance;
            return instance;
        }

        void SetTiming(Timing timing) {
            timing_ = timing;
        }

        Timing GetTiming() const {
            return timing_;
        }

    private:
        QnnMockSSRController() = default;
        QnnMockSSRController(const QnnMockSSRController&) = delete;
        QnnMockSSRController& operator=(const QnnMockSSRController&) = delete;

        // The timing to trigger SSR
        Timing timing_;
};
