// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

package ai.onnxruntime.example.javavalidator

import ai.onnxruntime.*
import ai.onnxruntime.OrtSession.SessionOptions
import android.os.Build;
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.microsoft.appcenter.espresso.Factory
import com.microsoft.appcenter.espresso.ReportHelper
import org.junit.*
import org.junit.runner.RunWith
import ai.onnxruntime.qnnpluginep.getEpName as getQnnPluginEpName
import ai.onnxruntime.qnnpluginep.getLibraryPath as getQnnPluginEpLibraryPath
import java.io.IOException
import java.util.*

private const val TAG = "ORTAndroidTest"
private const val QNN_EP_REGISTRATION_NAME = "QNNExecutionProvider"

@RunWith(AndroidJUnit4::class)
class SimpleTest {
    @get:Rule
    var reportHelper: ReportHelper = Factory.getReportHelper()

    @Before
    fun Start() {
        reportHelper.label("Starting App")
        Log.println(Log.INFO, TAG, "SystemABI=" + Build.SUPPORTED_ABIS[0])
    }

    @After
    fun TearDown() {
        reportHelper.label("Stopping App")
    }

    @Test
    fun runSigmoidModelTest() {
        for (intraOpNumThreads in 1..4) {
            runSigmoidModelTestImpl(intraOpNumThreads, OrtProvider.CPU)
        }
    }

    @Test
    fun runSigmoidModelTestQNN() {
        runSigmoidModelTestImpl(1, OrtProvider.QNN)
    }

    @Throws(IOException::class)
    private fun readModel(fileName: String): ByteArray {
        return InstrumentationRegistry.getInstrumentation().context.assets.open(fileName)
            .readBytes()
    }

    @Throws(OrtException::class, IOException::class)
    fun runSigmoidModelTestImpl(intraOpNumThreads: Int, executionProvider: OrtProvider) {
        reportHelper.label("Start Running Test with intraOpNumThreads=$intraOpNumThreads, executionProvider=$executionProvider")
        Log.println(Log.INFO, TAG, "Testing with intraOpNumThreads=$intraOpNumThreads")
        Log.println(Log.INFO, TAG, "Testing with executionProvider=$executionProvider")

        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)

        // The QNN EP shared library is not linked into ORT core; register it at runtime
        // so ORT can discover the QNNExecutionProvider.
        val qnnLibraryPath = getQnnPluginEpLibraryPath()
        env.registerExecutionProviderLibrary(QNN_EP_REGISTRATION_NAME, qnnLibraryPath)
        Log.println(Log.INFO, TAG, "registerExecutionProviderLibrary succeeded")

        // Unregister in a finally block so the library is always released even if the
        // test body throws (e.g. missing asset, session creation failure).  Without this,
        // a subsequent test invocation would fail with "already registered".
        // OrtEnvironment is a singleton; env.use {} above calls close() which decrements
        // its refcount but does not invalidate library management — unregister is safe here.
        try {
            env.use {
                val opts = SessionOptions()
                opts.setIntraOpNumThreads(intraOpNumThreads)

                when (executionProvider) {

                    OrtProvider.QNN -> {
                        val qnnEpName = getQnnPluginEpName()
                        val qnnDevices = env.epDevices.filter { it.epName == qnnEpName }
                        if (qnnDevices.isEmpty()) {
                            Log.println(Log.INFO, TAG, "NO QNN EP available, skip the test")
                            return
                        }
                        val providerOptions = Collections.singletonMap("backend_type", "htp")
                        opts.addExecutionProvider(qnnDevices, providerOptions)
                        opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                    }

                    OrtProvider.CPU -> {
                        // No additional configuration is needed for CPU
                    }

                    else -> {
                        //  Non exhaustive when statements on enum will be prohibited in future Gradle versions
                        Log.println(Log.INFO, TAG, "Skipping test as OrtProvider is not implemented")
                    }
                }

                opts.use {
                    val session = env.createSession(readModel("sigmoid.ort"), opts)
                    session.use {
                        val inputName = session.inputNames.iterator().next()
                        val testdata = Array(3) { Array(4) { FloatArray(5) } }
                        val expected = Array(3) { Array(4) { FloatArray(5) } }
                        for (i in 0..2) {
                            for (j in 0..3) {
                                for (k in 0..4) {
                                    testdata[i][j][k] = (i + j + k).toFloat()
                                    //expected sigmoid output is y = 1.0 / (1.0 + exp(-x))
                                    expected[i][j][k] =
                                        (1.0 / (1.0 + kotlin.math.exp(-testdata[i][j][k]))).toFloat()
                                }
                            }
                        }
                        val inputTensor = OnnxTensor.createTensor(env, testdata)
                        inputTensor.use {
                            val output = session.run(Collections.singletonMap(inputName, inputTensor))
                            output.use {
                                @Suppress("UNCHECKED_CAST")
                                val rawOutput = output[0].value as Array<Array<FloatArray>>
                                // QNN EP will run the Sigmoid float32 op with fp16 precision
                                val precision = if (executionProvider == OrtProvider.QNN) 1e-3 else 1e-6
                                for (i in 0..2) {
                                    for (j in 0..3) {
                                        for (k in 0..4) {
                                            Assert.assertEquals(
                                                rawOutput[i][j][k],
                                                expected[i][j][k],
                                                precision.toFloat()
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            runCatching { env.unregisterExecutionProviderLibrary(QNN_EP_REGISTRATION_NAME) }
                .onFailure { Log.w("SimpleTest", "unregisterExecutionProviderLibrary failed", it) }
        }
    }
}
