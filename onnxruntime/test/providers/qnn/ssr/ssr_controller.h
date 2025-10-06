class QnnSSRController {
    public:
        enum class Timing {
            GraphFinalize,
            GraphExecute
        };

        static QnnSSRController& Instance() {
            static QnnSSRController instance;
            return instance;
        }

        void SetTiming(Timing timing) {
            timing_ = timing;
        }

        Timing GetTiming() const {
            return timing_;
        }

    private:
        QnnSSRController() = default;
        QnnSSRController(const QnnSSRController&) = delete;
        QnnSSRController& operator=(const QnnSSRController&) = delete;

        // The timing to trigger SSR
        Timing timing_;
};
