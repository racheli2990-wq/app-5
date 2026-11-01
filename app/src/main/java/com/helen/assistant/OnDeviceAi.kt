package com.helen.assistant

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.net.Uri
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * Owns the LiteRT-LM engine and all conversations on one executor.
 * Model weights are bundled in the APK and copied once into app-private storage.
 */
class OnDeviceAi(context: Context) : AutoCloseable {
    interface ModelListener {
        fun onProgress(percent: Int, status: String)
        fun onReady()
        fun onError(message: String)
    }
    interface AiCallback {
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
    private val modelFile = File(modelDir, "qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm")
    @Volatile private var ready = false
    private var engine: Engine? = null
    private var chat: Conversation? = null
    private var completion: Conversation? = null

    companion object {
        const val MODEL_BYTES_APPROX = 347_000_000L
        const val MODEL_URL =
            "https://huggingface.co/litert-community/Qwen3-0.6B-int4/resolve/main/qwen3_0.6b_nothink_q4_block32_ekv1280.litertlm?download=true"
        private const val MIN_VALID_BYTES = 300_000_000L
        private const val SYSTEM_PROMPT = """
You are HELEN, a Hebrew-first personal assistant running locally on an Android phone.
Always understand natural Hebrew, colloquial phrases, incomplete wording and context from prior turns.
Return ONLY one valid JSON object, without markdown and without text outside JSON:
{"reply":"short natural Hebrew response","actions":[{"tool":"tool_name","argument":"value"}]}
For an ordinary question or conversation, use an empty actions array and answer in Hebrew.
You may return several actions for one sentence. Never invent that an action succeeded.
Available tools:
- flashlight: enabled (boolean)
- open_app: app (string, app name)
- camera: no arguments
- dial: number (string). Open dialer only, never place a call.
- whatsapp: message (string, optional). Open WhatsApp/share confirmation only, never send automatically.
- volume: direction ("up" or "down") OR percent (integer 0..100)
- brightness: percent (integer 0..100); the app may ask for Android permission.
- ringer: mode ("normal" or "silent"). Respect Android permission rules.
- wifi_settings, bluetooth_settings: no arguments; open the Android settings/panel.
- scroll: direction ("up" or "down"); requires user-enabled accessibility service.
- tap: x and y screen coordinates; only when user explicitly asks and accessibility is enabled.
- type_text: text; only in focused field and when accessibility is enabled.
- back, home, screenshot: no arguments; requires accessibility service.
- reminder: minutes (integer) and text (string).
- timer: seconds (integer) and text (string).
- alarm: hour (0..23), minute (0..59), text (string).
- note_add, task_add, shopping_add: text (string).
- expense_add: amount (number).
- water_add: cups (integer, default 1).
- web_search: query (string), for external lookup only when explicitly requested.
- transit_search: query (string), open a public transit/maps search.
- knowledge_search: query (string), search the user's imported local text file.
- hebrew_date: no arguments.
Important safety and accuracy rules:
- If a request needs a missing essential detail (such as phone number, reminder time, or text), ask one concise question and do not emit an action.
- Never send messages, place calls, buy, delete, or change a security setting by itself. Open the relevant app/confirmation screen instead.
- For Wi-Fi and Bluetooth, open settings; Android does not allow ordinary apps to silently toggle these radios.
- Don't claim to control other apps' private UI unless the accessibility service is enabled.
- Keep responses brief, helpful, warm and non-childish. Use the conversation context for pronouns such as "turn it off".
"""
    }

    fun isReady(): Boolean = ready
    fun hasModel(): Boolean = modelFile.exists() && modelFile.length() >= MIN_VALID_BYTES
    fun modelPath(): String = modelFile.absolutePath

    fun prepare(listener: ModelListener) {
        worker.execute {
            try {
                if (!hasModel()) {
                    try {
                        app.assets.open(modelFile.name).use { input ->
                            modelFile.outputStream().buffered().use { output ->
                                val buffer = ByteArray(64 * 1024)
                                var copied = 0L
                                var read: Int
                                while (input.read(buffer).also { read = it } != -1) {
                                    output.write(buffer, 0, read)
                                    copied += read
                                    postProgress(listener, ((copied * 100) / MODEL_BYTES_APPROX).toInt().coerceAtMost(99),
                                        "מכינה את מודל ה־AI המובנה… ${formatMb(copied)} MB")
                                }
                            }
                        }
                    } catch (_: Throwable) {
                        if (!hasModel()) downloadModel(listener)
                    }
                } else {
                    postProgress(listener, 100, "המודל המקומי נמצא. טוענת את המנוע…")
                }
                if (!hasModel()) throw IllegalStateException("קובץ מודל ה־AI חסר או לא תקין")
                initializeModel()
                ready = true
                main.post { listener.onReady() }
            } catch (t: Throwable) {
                ready = false
                main.post { listener.onError(t.message ?: "טעינת מודל ה־AI נכשלה") }
            }
        }
    }

    private fun downloadModel(listener: ModelListener) {
        postProgress(listener, 0, "מורידה מודל AI מקומי (כ־475 MiB)…")
        val part = File(modelDir, modelFile.name + ".part")
        if (part.exists()) part.delete()
        val connection = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 180_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Helen-Android/2.0")
            setRequestProperty("Accept-Encoding", "identity")
        }
        try {
            connection.connect()
            val code = connection.responseCode
            if (code !in 200..299) throw IllegalStateException("שרת המודל החזיר HTTP $code")
            val total = connection.contentLengthLong
            var received = 0L
            connection.inputStream.use { input ->
                part.outputStream().buffered().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var read: Int
                    var lastPercent = -1
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        received += read
                        val percent = if (total > 0) ((received * 100) / total).toInt() else -1
                        if (percent != lastPercent) {
                            lastPercent = percent
                            postProgress(listener, percent, "מורידה מודל AI… ${formatMb(received)} MB")
                        }
                    }
                    output.flush()
                }
            }
            if (received < MIN_VALID_BYTES) throw IllegalStateException("קובץ המודל שהתקבל קטן מדי או פגום")
            if (modelFile.exists()) modelFile.delete()
            if (!part.renameTo(modelFile)) {
                part.copyTo(modelFile, overwrite = true)
                part.delete()
            }
            postProgress(listener, 100, "המודל ירד. מאתחלת AI מקומי…")
        } finally {
            connection.disconnect()
            if (part.exists()) part.delete()
        }
    }

    private fun formatMb(bytes: Long): String = String.format("%.0f", bytes / (1024.0 * 1024.0))
    private fun postProgress(listener: ModelListener, percent: Int, status: String) {
        main.post { listener.onProgress(percent, status) }
    }

    private fun initializeModel() {
        val config = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = Backend.CPU(),
            cacheDir = File(app.cacheDir, "litert-cache").apply { mkdirs() }.absolutePath
        )
        val candidate = Engine(config)
        candidate.initialize()
        val sampler = SamplerConfig(topK = 32, topP = 0.9, temperature = 0.25)
        val chatConversation = candidate.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(SYSTEM_PROMPT),
                samplerConfig = sampler
            )
        )
        val completionConversation = candidate.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(
                    "את מנוע השלמת מילים בעברית. קבלי טקסט חלקי והחזירי בדיוק 3 המשכים טבעיים וקצרים בעברית, " +
                    "כל אפשרות בשורה נפרדת, בלי מספור, בלי הסברים ובלי מרכאות. השלימי את המשפט של המשתמש, " +
                    "אל תעני על הפקודה עצמה. אם אין מספיק הקשר, החזירי 2 הצעות קצרות."
                ),
                samplerConfig = SamplerConfig(topK = 24, topP = 0.88, temperature = 0.35)
            )
        )
        engine = candidate
        chat = chatConversation
        completion = completionConversation
    }

    fun ask(userText: String, callback: AiCallback) {
        worker.execute {
            try {
                val conversation = chat ?: throw IllegalStateException("מודל ה־AI עדיין לא מוכן")
                val answer = conversation.sendMessage(userText).text
                main.post { callback.onSuccess(answer ?: "") }
            } catch (t: Throwable) {
                main.post { callback.onError(t.message ?: "שגיאה בהרצת מודל ה־AI") }
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
                val prompt = "השלימי את הפקודה או המשפט הבא ותני הצעות שימושיות:\n$partialText"
                val raw = completion?.sendMessage(prompt)?.text.orEmpty()
                val suggestions = raw
                    .replace("", "")
                    .lineSequence()
                    .map { it.trim().removePrefix("-").trim().removePrefix("•").trim() }
                    .filter { it.isNotBlank() && it.length <= 100 && !it.startsWith("{") }
                    .distinct()
                    .take(3)
                    .toList()
                main.post { callback.onSuggestions(suggestions) }
            } catch (_: Throwable) {
                main.post { callback.onSuggestions(emptyList()) }
            }
        }
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
