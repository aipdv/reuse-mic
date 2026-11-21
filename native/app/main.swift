// ReuseMic: menu bar front end for the native `reuse live` engine (NVIDIA RE-USE, on-device).
// Mic -> RE-USE -> BlackHole 2ch. Pick "BlackHole 2ch" as the microphone in the meeting app.
import AppKit

struct Quality { let title: String; let args: [String] }
let qualities = [
    Quality(title: "Light: 16 kHz, 4 layers (about 0.4 core)", args: ["--rate", "16000", "--layers", "4", "--threads", "1"]),
    Quality(title: "High: 16 kHz, 8 layers, 2 threads (about 0.9 core)", args: ["--rate", "16000", "--layers", "8", "--threads", "2"]),
    Quality(title: "Wide band: 32 kHz, 4 layers, 2 threads (about 0.95 core)", args: ["--rate", "32000", "--layers", "4", "--threads", "2"]),
    Quality(title: "Full band: 48 kHz, 4 layers, 4 threads (about 1.8 cores)", args: ["--rate", "48000", "--layers", "4", "--threads", "4"]),
]

final class App: NSObject, NSApplicationDelegate {
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    var proc: Process?
    var stdinPipe: Pipe?
    var enhanced = true
    var quality = UserDefaults.standard.integer(forKey: "quality")
    var mic = UserDefaults.standard.string(forKey: "mic")  // nil = system default input
    var lastLog = ""

    var engine: URL { Bundle.main.resourceURL!.appendingPathComponent("reuse") }

    func applicationDidFinishLaunching(_ n: Notification) { refresh() }
    func applicationWillTerminate(_ n: Notification) { stop() }

    func inputDevices() -> [String] {
        let p = Process(); p.executableURL = engine; p.arguments = ["devices"]
        let out = Pipe(); p.standardOutput = out
        guard (try? p.run()) != nil else { return [] }
        p.waitUntilExit()
        let text = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return text.split(separator: "\n").compactMap { line in
            // "<name padded to 40> in  N  out ..."
            guard let r = line.range(of: " in ", options: .backwards) else { return nil }
            let name = line[..<r.lowerBound].trimmingCharacters(in: .whitespaces)
            let rest = line[r.upperBound...].trimmingCharacters(in: .whitespaces)
            let ch = Int(rest.split(separator: " ").first ?? "0") ?? 0
            return ch > 0 && !name.contains("BlackHole") ? name : nil
        }
    }

    func refresh() {
        let running = proc?.isRunning == true
        let symbol = !running ? "waveform.slash" : (enhanced ? "waveform" : "waveform.path")
        item.button?.image = NSImage(systemSymbolName: symbol, accessibilityDescription: "ReuseMic")
        let m = NSMenu()
        let status = NSMenuItem(title: running ? (enhanced ? "Running: enhanced" : "Running: raw (A/B)") : "Stopped", action: nil, keyEquivalent: "")
        status.isEnabled = false; m.addItem(status)
        m.addItem(.separator())
        m.addItem(withTitle: running ? "Stop" : "Start", action: #selector(toggleRun), keyEquivalent: "s").target = self
        let e = m.addItem(withTitle: "Enhanced", action: #selector(toggleEnhanced), keyEquivalent: "e")
        e.target = self; e.state = enhanced ? .on : .off; e.isEnabled = running
        m.addItem(.separator())
        let qm = NSMenu()
        for (i, q) in qualities.enumerated() {
            let it = qm.addItem(withTitle: q.title, action: #selector(pickQuality(_:)), keyEquivalent: "")
            it.target = self; it.tag = i; it.state = i == quality ? .on : .off
        }
        m.addItem(withTitle: "Quality", action: nil, keyEquivalent: "").submenu = qm
        let mm = NSMenu()
        let def = mm.addItem(withTitle: "System default", action: #selector(pickMic(_:)), keyEquivalent: "")
        def.target = self; def.state = mic == nil ? .on : .off
        for d in inputDevices() {
            let it = mm.addItem(withTitle: d, action: #selector(pickMic(_:)), keyEquivalent: "")
            it.target = self; it.representedObject = d; it.state = d == mic ? .on : .off
        }
        m.addItem(withTitle: "Microphone", action: nil, keyEquivalent: "").submenu = mm
        let out = NSMenuItem(title: "Output: BlackHole 2ch (pick it as the mic in your call app)", action: nil, keyEquivalent: "")
        out.isEnabled = false; m.addItem(out)
        if !lastLog.isEmpty { let l = NSMenuItem(title: lastLog, action: nil, keyEquivalent: ""); l.isEnabled = false; m.addItem(l) }
        m.addItem(.separator())
        m.addItem(withTitle: "Quit", action: #selector(quit), keyEquivalent: "q").target = self
        item.menu = m
    }

    func start() {
        let p = Process(); p.executableURL = engine
        var args = ["live", "--output", "BlackHole 2ch"] + qualities[min(quality, qualities.count - 1)].args
        if let mic { args += ["--input", mic] }
        p.arguments = args
        let inp = Pipe(), out = Pipe(); p.standardInput = inp; p.standardOutput = out; p.standardError = out
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let s = String(data: h.availableData, encoding: .utf8) ?? ""
            if let line = s.split(separator: "\n").last(where: { $0.hasPrefix("hops") || $0.contains("error") || $0.contains("not found") }) {
                DispatchQueue.main.async { self?.lastLog = String(line.prefix(90)); self?.refresh() }
            }
        }
        p.terminationHandler = { [weak self] _ in DispatchQueue.main.async { self?.refresh() } }
        do { try p.run() } catch { lastLog = "cannot start engine: \(error.localizedDescription)"; refresh(); return }
        proc = p; stdinPipe = inp; enhanced = true; lastLog = ""
        refresh()
    }
    func stop() {
        guard let p = proc, p.isRunning else { return }
        p.interrupt()  // SIGINT: the engine prints its stats and exits cleanly
        p.waitUntilExit(); proc = nil
    }

    @objc func toggleRun() { if proc?.isRunning == true { stop() } else { start() }; refresh() }
    @objc func toggleEnhanced() {
        stdinPipe?.fileHandleForWriting.write("\n".data(using: .utf8)!)  // the engine toggles on Enter
        enhanced.toggle(); refresh()
    }
    @objc func pickQuality(_ s: NSMenuItem) {
        quality = s.tag; UserDefaults.standard.set(quality, forKey: "quality")
        if proc?.isRunning == true { stop(); start() } else { refresh() }
    }
    @objc func pickMic(_ s: NSMenuItem) {
        mic = s.representedObject as? String; UserDefaults.standard.set(mic, forKey: "mic")
        if proc?.isRunning == true { stop(); start() } else { refresh() }
    }
    @objc func quit() { stop(); NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
