package com.helen.assistant

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Local Gemma 4 runtime. The large weight file is downloaded once and kept offline. */
class OnDeviceAi(context: Context) : AutoCloseable {
    interface ModelListener {
        fun onProgress(percent: Int, status: String)
        fun onReady()
        fun onError(message: String)
    }
    interface AiCallback {
        fun onProgress()
        fun onSuccess(text: String)
        fun onError(message: String)
    }
    interface SuggestionCallback {
        fun onSuggestions(items: List<String>)
    }

    private val app = context.applicationContext
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val modelDir = File(app.filesDir, "models").apply { mkdirs() }
    private val modelFile = File(modelDir, MODEL_NAME)
    private val partialFile = File(modelDir, "$MODEL_NAME.part")
    @Volatile private var ready = false
    @Volatile private var activeBackend = "בהכנה"
    private var engine: Engine? = null
    private var chat: Conversation? = null
    private var completion: Conversation? = null

    companion object {
        const val MODEL_NAME = "gemma-4-E2B-it.litertlm"
        const val MODEL_BYTES_APPROX = 2_583_000_000L
        const val MODEL_URL = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm?download=true"
        private const val MIN_VALID_BYTES = 2_400_000_000L
        private const val SYSTEM_PROMPT = """
You are HELEN, a Hebrew-first personal assistant running locally on Android.
Understand modern Hebrew, colloquial speech, typos, incomplete sentences, pronouns and prior conversation context.
Reply naturally in Hebrew unless the user uses another language. Be clear, accurate and concise.

For every response return ONLY one valid JSON object:
{"reply":"short natural Hebrew response","actions":[{"tool":"tool_name","args":{"key":"value"}}]}
For ordinary questions or conversation use "actions":[].
A single sentence may request multiple actions. Use only tools listed below.
Never claim an action succeeded until the app confirms it.

TOOLS:
- flashlight: {"enabled":true|false}
- open_app: {"app":"application name"}
- camera: {}
- dial: {"number":"phone number"} (open dialer only; never place a call)
- whatsapp: {"message":"optional text"} (open WhatsApp/share confirmation only; never send silently)
- volume: {"direction":"up"|"down"} or {"percent":0..100}
- brightness: {"percent":0..100}
- ringer: {"mode":"normal"|"silent"}
- wifi_settings, bluetooth_settings: {}
- scroll: {"direction":"up"|"down"} (accessibility service required)
- tap: {"x":number,"y":number} (only on explicit request; accessibility required)
- type_text: {"text":"text"} (focused field and accessibility required)
- back, home, screenshot, read_screen: {}
- reminder: {"minutes":integer,"text":"reminder"}
- timer: {"seconds":integer,"text":"label"}
- alarm: {"hour":0..23,"minute":0..59,"text":"label"}
- note_add, task_add, shopping_add: {"text":"content"}
- expense_add: {"amount":number}
- water_add: {"cups":integer}
- calendar_add: {"title":"event","minutes":integer from now}
- web_search, transit_search: {"query":"search phrase"} (only when explicitly requested)
- knowledge_search: {"query":"phrase in imported local file"}
- hebrew_date: {}

RULES:
- If an essential detail is missing, ask one short question and emit no action.
- Never send a message, place a call, delete content, purchase anything or alter a security setting silently.
- For Wi-Fi/Bluetooth, open Android settings; do not claim the radios were toggled.
- Brightness and silent mode may require explicit Android permission.
- Screen taps, typing, scrolling and screenshots require the user's explicit accessibility opt-in.
- If a service is unavailable, explain which permission is needed.
- Keep replies concise and don't invent details.
"""
    }

    fun isReady(): Boolean = ready
    fun backendName(): String = activeBackend
    fun hasModel(): Boolean = modelFile.exists() && modelFile.length() >= MIN_VALID_BYTES
    fun modelPath(): String = modelFile.absolutePath
    fun partialBytes(): Long = if (partialFile.exists()) partialFile.length() else 0L

    fun prepare(listener: ModelListener) {
        worker.execute {
            try {
                if (!hasModel()) downloadModel(listener)
                postProgress(listener, 100, "Gemma 4 הורד. מאתחלת מנוע AI…")
                if (!hasModel()) throw IllegalStateException("קובץ המודל לא תקין")
                initializeModel()
                ready = true
                main.post { listener.onReady() }
            } catch (t: Throwable) {
                ready = false
                main.post { listener.onError(t.message ?: "טעינת מודל AI נכשלה") }
            }
        }
    }

    private fun downloadModel(listener: ModelListener) {
        var existing = if (partialFile.exists()) partialFile.length() else 0L
        if (existing > MODEL_BYTES_APPROX) {
            partialFile.delete()
            existing = 0L
        }
        val needed = (MODEL_BYTES_APPROX - existing).coerceAtLeast(0L) + 500_000_000L
        if (modelDir.usableSpace < needed) {
            throw IllegalStateException("אין מספיק מקום פנוי. נדרשים עוד כ־${formatMb(needed)}MB פנויים להשלמת ההורדה וההכנה.")
        }
        postProgress(listener, if (existing > 0) ((existing * 100) / MODEL_BYTES_APPROX).toInt() else 0,
            if (existing > 0) "ממשיכה הורדת Gemma 4 מהמקום שבו נעצרה…" else "מורידה Gemma 4 · כ־2.58GB")
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 240_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Helen-AI-3.0")
            setRequestProperty("Accept-Encoding", "identity")
            if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
        }
        try {
            connection.connect()
            val code = connection.responseCode
            if (code !in 200..299) throw IllegalStateException("שרת המודל החזיר HTTP $code")
            val append = existing > 0 && code == HttpURLConnection.HTTP_PARTIAL
            if (existing > 0 && !append) {
                partialFile.delete()
                existing = 0L
            }
            val total = if (connection.contentLengthLong > 0) existing + connection.contentLengthLong else MODEL_BYTES_APPROX
            var received = existing
            val output = if (append) {
                java.io.FileOutputStream(partialFile, true).buffered(1024 * 1024)
            } else {
                partialFile.outputStream().buffered(1024 * 1024)
            }
            connection.inputStream.use { input ->
                output.use { out ->
                    val buffer = ByteArray(1024 * 1024)
                    var read: Int
                    var lastPercent = -1
                    while (input.read(buffer).also { read = it } != -1) {
                        out.write(buffer, 0, read)
                        received += read
                        val percent = ((received * 100L) / total).toInt().coerceIn(0, 99)
                        if (percent != lastPercent) {
                            lastPercent = percent
                            postProgress(listener, percent, "מורידה מודל AI · ${formatMb(received)}MB / כ־2,583MB")
                        }
                    }
                    out.flush()
                }
            }
            if (partialFile.length() < MIN_VALID_BYTES) {
                throw IllegalStateException("ההורדה הסתיימה מוקדם מדי. אפשר לנסות שוב וההורדה תמשיך.")
            }
            if (modelFile.exists()) modelFile.delete()
            if (!partialFile.renameTo(modelFile)) {
                partialFile.copyTo(modelFile, overwrite = true)
                partialFile.delete()
            }
        } finally {
            connection.disconnect()
        }
    }

    @OptIn(ExperimentalApi::class)
    private fun initializeModel() {
        ExperimentalFlags.enableSpeculativeDecoding = true
        val cache = File(app.cacheDir, "gemma4-cache").apply { mkdirs() }
        var selected: Engine? = null
        var gpuReady = false
        try {
            val gpu = Engine(EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                cacheDir = cache.absolutePath
            ))
            gpu.initialize()
            selected = gpu
            gpuReady = true
            activeBackend = "GPU מואץ + MTP"
        } catch (_: Throwable) {
            // Some phones do not expose a compatible OpenCL driver. Retry with CPU.
        }
        if (!gpuReady) {
            val cpu = Engine(EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                cacheDir = File(app.cacheDir, "gemma4-cpu-cache").apply { mkdirs() }.absolutePath
            ))
            cpu.initialize()
            selected = cpu
            activeBackend = "CPU + MTP"
        }
        val sampler = SamplerConfig(topK = 24, topP = 0.88, temperature = 0.2)
        try {
            val chatConversation = selected!!.createConversation(ConversationConfig(
                systemInstruction = Contents.of(SYSTEM_PROMPT),
                samplerConfig = sampler
            ))
            val completionConversation = selected!!.createConversation(ConversationConfig(
                systemInstruction = Contents.of(
                    "את מנוע השלמת המשפטים של הלן. החזירי 3 המשכים טבעיים וקצרים בעברית, " +
                    "אפשרות אחת בכל שורה, ללא מספור או הסברים. השלימי את הניסוח ואל תעני לפקודה עצמה."
                ),
                samplerConfig = SamplerConfig(topK = 20, topP = 0.85, temperature = 0.35)
            ))
            engine = selected
            chat = chatConversation
            completion = completionConversation
        } catch (t: Throwable) {
            try { selected?.close() } catch (_: Throwable) {}
            throw t
        }
    }

    fun ask(userText: String, callback: AiCallback) {
        worker.execute {
            try {
                val conversation = chat ?: throw IllegalStateException("מודל ה־AI עדיין לא מוכן")
                val result = streamMessage(conversation, userText) { callback.onProgress() }
                main.post { callback.onSuccess(result) }
            } catch (t: Throwable) {
                main.post { callback.onError(t.message ?: "שגיאה בהרצת Gemma 4") }
            }
        }
    }

    fun suggest(partialText: String, callback: SuggestionCallback) {
        if (!ready || partialText.trim().length < 3) {
            main.post { callback.onSuggestions(emptyList()) }
            return
        }
        worker.execute {
            try {
                val prompt = "השלימי את המשפט או הפקודה הבאים:\n$partialText"
                val raw = streamMessage(completion ?: throw IllegalStateException("מנוע ההשלמה אינו זמין"), prompt)
                val items = raw.replace("\r", "").lineSequence()
                    .map { it.trim().removePrefix("-").removePrefix("•").trim() }
                    .filter { it.isNotBlank() && it.length <= 90 && !it.startsWith("{") }
                    .distinct().take(3).toList()
                main.post { callback.onSuggestions(items) }
            } catch (_: Throwable) {
                main.post { callback.onSuggestions(emptyList()) }
            }
        }
    }

    private fun streamMessage(conversation: Conversation, prompt: String, onChunk: (() -> Unit)? = null): String {
        val output = StringBuilder()
        val error = AtomicReference<Throwable?>(null)
        val latch = CountDownLatch(1)
        conversation.sendMessageAsync(prompt, object : MessageCallback {
            override fun onMessage(message: Message) {
                output.append(message.toString())
                onChunk?.invoke()
            }
            override fun onDone() { latch.countDown() }
            override fun onError(throwable: Throwable) {
                error.set(throwable)
                latch.countDown()
            }
        })
        if (!latch.await(5, TimeUnit.MINUTES)) throw IllegalStateException("המודל לא סיים בזמן. נסי שוב.")
        error.get()?.let { throw it }
        return output.toString()
    }

    private fun formatMb(bytes: Long): String = String.format("%.0f", bytes / (1024.0 * 1024.0))
    private fun postProgress(listener: ModelListener, percent: Int, status: String) {
        main.post { listener.onProgress(percent, status) }
    }

    override fun close() {
        ready = false
        worker.execute {
            try { completion?.close() } catch (_: Throwable) {}
            try { chat?.close() } catch (_: Throwable) {}
            try { engine?.close() } catch (_: Throwable) {}
            completion = null
            chat = null
            engine = null
        }
        worker.shutdown()
    }
}
