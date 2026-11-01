package com.helen.assistant

import android.content.Context
import android.app.ActivityManager
import android.os.Handler
import android.os.Looper
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger

/** Local Qwen3 1.7B runtime. The GPU-optimized INT4 model ships inside the APK. */
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
    private var restoredHistory: List<String> = emptyList()
    private val pendingChatRequests = AtomicInteger(0)

    companion object {
        const val MODEL_NAME = "Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm"
        const val MODEL_BYTES_APPROX = 980_000_000L
        private const val MIN_VALID_BYTES = 850_000_000L
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
    fun modelSizeBytes(): Long = if (modelFile.exists()) modelFile.length() else partialBytes()

    fun setConversationHistory(messages: List<String>) {
        restoredHistory = messages.takeLast(12).toList()
    }

    fun prepare(listener: ModelListener) {
        worker.execute {
            try {
                if (ready) {
                    main.post { listener.onReady() }
                    return@execute
                }
                if (!hasModel()) installBundledModel(listener)
                postProgress(listener, 100, "מודל Qwen3 1.7B המובנה מוכן. מאתחלת מנוע AI…")
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

    /** Copies the model bundled in APK assets; this never uses the network. */
    private fun installBundledModel(listener: ModelListener) {
        File(modelDir, "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm").delete()
        File(modelDir, "qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm").delete()
        File(app.cacheDir, "qwen25-cache").deleteRecursively()
        File(app.cacheDir, "qwen25-cpu-cache").deleteRecursively()
        if (modelDir.usableSpace < MODEL_BYTES_APPROX + 150_000_000L) {
            throw IllegalStateException("אין מספיק מקום פנוי להעתקת מודל ה־AI המובנה. נדרשים כ־1.1GB פנויים.")
        }
        postProgress(listener, 0, "מעתיקה את מודל Qwen3 1.7B מתוך קובץ ההתקנה…")
        val installingFile = File(modelDir, "$MODEL_NAME.installing")
        try {
            app.assets.open(MODEL_NAME).buffered(1024 * 1024).use { input ->
                FileOutputStream(installingFile).buffered(1024 * 1024).use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var received = 0L
                    var lastPercent = -1
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        received += read
                        val percent = ((received * 100L) / MODEL_BYTES_APPROX).toInt().coerceIn(0, 99)
                        if (percent != lastPercent) {
                            lastPercent = percent
                            postProgress(listener, percent, "מכינה מודל AI מובנה · ${formatMb(received)}MB")
                        }
                    }
                    output.flush()
                }
            }
            if (installingFile.length() < MIN_VALID_BYTES) {
                throw IllegalStateException("קובץ מודל ה־AI שבתוך ה־APK חסר או פגום.")
            }
            if (modelFile.exists() && !modelFile.delete()) {
                throw IllegalStateException("לא ניתן להחליף את קובץ המודל הקודם.")
            }
            if (!installingFile.renameTo(modelFile)) {
                installingFile.copyTo(modelFile, overwrite = true)
                installingFile.delete()
            }
            File(modelDir, "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm").delete()
            File(modelDir, "qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm").delete()
            File(app.cacheDir, "qwen25-cache").deleteRecursively()
            File(app.cacheDir, "qwen25-cpu-cache").deleteRecursively()
            File(app.cacheDir, "qwen3-cache").deleteRecursively()
            File(app.cacheDir, "qwen3-cpu-cache").deleteRecursively()
            postProgress(listener, 100, "מודל Qwen3 1.7B הותקן מתוך ה־APK.")
        } catch (t: Throwable) {
            installingFile.delete()
            throw t
        }
    }

    private fun initializeModel() {
        val cache = File(app.cacheDir, "qwen3-17b-cache").apply { mkdirs() }
        var selected: Engine? = null
        var gpuEngine: Engine? = null
        var gpuReady = false
        try {
            gpuEngine = Engine(EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                cacheDir = cache.absolutePath
            ))
            gpuEngine.initialize()
            selected = gpuEngine
            gpuReady = true
            activeBackend = "GPU מואץ"
        } catch (_: Throwable) {
            try { gpuEngine?.close() } catch (_: Throwable) {}
        }
        if (!gpuReady) {
            val activityManager = app.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            if (memoryInfo.availMem < 2_000_000_000L) {
                throw IllegalStateException("הטלפון לא סיפק מספיק זיכרון פנוי להפעלת מודל 1.7B במעבד. מומלץ לסגור אפליקציות אחרות או להשתמש במכשיר עם האצת GPU וזיכרון פנוי רב יותר.")
            }
            val cpu = Engine(EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.CPU(),
                cacheDir = File(app.cacheDir, "qwen3-17b-cpu-cache").apply { mkdirs() }.absolutePath
            ))
            cpu.initialize()
            selected = cpu
            activeBackend = "CPU (איטי יותר · זיכרון גבוה)"
        }
        val sampler = SamplerConfig(topK = 20, topP = 0.95, temperature = 0.6)
        try {
            val restoredMessages = restoredHistory.mapNotNull { item ->
                when {
                    item.startsWith("את:") -> Message.user(item.removePrefix("את:").trim())
                    item.startsWith("הלן:") -> Message.model(item.removePrefix("הלן:").trim())
                    else -> null
                }
            }
            val chatConversation = selected!!.createConversation(ConversationConfig(
                systemInstruction = Contents.of(SYSTEM_PROMPT),
                initialMessages = restoredMessages,
                samplerConfig = sampler,
                thinkingConfig = ThinkingConfig(enableThinking = true, thinkingTokenBudget = 160),
                maxOutputToken = 320
            ))
            val completionConversation = selected!!.createConversation(ConversationConfig(
                systemInstruction = Contents.of(
                    "את מנוע השלמת המשפטים של הלן. החזירי 3 המשכים טבעיים וקצרים בעברית, " +
                    "אפשרות אחת בכל שורה, ללא מספור או הסברים. השלימי את הניסוח ואל תעני לפקודה עצמה."
                ),
                samplerConfig = SamplerConfig(topK = 20, topP = 0.8, temperature = 0.55),
                thinkingConfig = ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0),
                maxOutputToken = 64
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
        pendingChatRequests.incrementAndGet()
        worker.execute {
            try {
                val conversation = chat ?: throw IllegalStateException("מודל ה־AI עדיין לא מוכן")
                val result = streamMessage(conversation, userText, 320) { callback.onProgress() }
                main.post { callback.onSuccess(result) }
            } catch (t: Throwable) {
                main.post { callback.onError(t.message ?: "שגיאה בהרצת מודל Qwen3 1.7B") }
            } finally {
                pendingChatRequests.decrementAndGet()
            }
        }
    }

    fun suggest(partialText: String, callback: SuggestionCallback) {
        if (!ready || pendingChatRequests.get() > 0 || partialText.trim().length < 3) {
            main.post { callback.onSuggestions(emptyList()) }
            return
        }
        worker.execute {
            try {
                if (pendingChatRequests.get() > 0) {
                    main.post { callback.onSuggestions(emptyList()) }
                    return@execute
                }
                val prompt = "השלימי את המשפט או הפקודה הבאים:\n$partialText"
                val raw = streamMessage(
                    completion ?: throw IllegalStateException("מנוע ההשלמה אינו זמין"),
                    prompt,
                    64
                )
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

    private fun streamMessage(
        conversation: Conversation,
        prompt: String,
        maxOutputTokens: Int,
        onChunk: (() -> Unit)? = null
    ): String {
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
        }, maxOutputToken = maxOutputTokens)
        if (!latch.await(5, TimeUnit.MINUTES)) throw IllegalStateException("המודל לא סיים בזמן. נסי שוב.")
        error.get()?.let { throw it }
        return output.toString()
    }

    fun deleteModel(callback: java.util.function.Consumer<Boolean>) {
        worker.execute {
            ready = false
            try { completion?.close() } catch (_: Throwable) {}
            try { chat?.close() } catch (_: Throwable) {}
            try { engine?.close() } catch (_: Throwable) {}
            completion = null
            chat = null
            engine = null
            val partialDeleted = !partialFile.exists() || partialFile.delete()
            val modelDeleted = !modelFile.exists() || modelFile.delete()
            activeBackend = "לא מותקן"
            main.post { callback.accept(partialDeleted && modelDeleted) }
        }
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
