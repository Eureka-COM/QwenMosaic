# Colab で、ある計画が終わったら同じランタイムのまま次の計画を始める

Colab の G4 は取りにくい。runner は実行の終わりにランタイムを返す（auto_stop）ので、そのままだと次の段でまた取り直しになる。
走っている実行はそのままに、次の 2 つを足す（2026-11-12、使い捨ての CPU ノートブックで再現試験済み）。

## (a) VM の端末の見張り：ランタイムを返す手前でセルを止める

Colab の「ターミナル」で（`774` はカーネルの PID：`pgrep -f colab_kernel_launcher`。`s2` は終わりを待つ run の名前）：

```bash
nohup bash -c 'while true; do if grep -q finished /content/out/status.json 2>/dev/null; then if grep -q "s2\": \"ok" /content/out/status.json; then kill -INT 774; sleep 2; kill -INT 774; date -u > /content/keep.log; fi; break; fi; sleep 0.2; done' >/dev/null 2>&1 &
```

runner は全部の run が終わると status.json に finished を書き、最後の保存（数秒かかる）をしてからランタイムを返す。
その保存の最中に割り込みが入り、セルは KeyboardInterrupt で終わって、ランタイムは残る。run が失敗で終わったときは何もしない（ランタイムは返される）。

端末への入力は、ページの JS から `document.querySelector('colab-xterm-terminal').terminal.paste(...)` で貼り、Return キーを送る。

## (b) ページの見張り：止まったら次の計画でセルを実行する

Colab のタブのページ内で（タブを再読み込みすると消えるので、入れ直す）：

```js
(() => { if (window.__mqChain) clearInterval(window.__mqChain); window.__mqLog = []; window.__mqSeen = false; window.__mqDone = false;
  const FROM = 'colab/plans/mq-27b-s2.json', TO = 'colab/plans/mq-27b-s3.json';
  const connText = () => { const c = document.querySelector('colab-connect-button'); return c && c.shadowRoot ? c.shadowRoot.textContent.replace(/\s+/g, ' ').trim() : ''; };
  const isRunning = () => !!document.querySelector('.cell.running, .cell.pending');
  window.__mqChain = setInterval(() => {
    if (isRunning()) { window.__mqSeen = true; return; }
    if (!window.__mqSeen || window.__mqDone) return;
    window.__mqDone = true; clearInterval(window.__mqChain);
    window.__mqLog.push('stopped ' + new Date().toISOString() + ' conn=' + connText());
    setTimeout(() => {
      const ct = connText(); const m = monaco.editor.getEditors()[0].getModel(); const v = m.getValue();
      if (isRunning()) { window.__mqLog.push('already running again: no chain'); return; }
      if (!/done/.test(ct) || /再接続|Reconnect/.test(ct)) { window.__mqLog.push('runtime not connected: no chain. conn=' + ct); return; }
      if (!v.includes(FROM)) { window.__mqLog.push('plan line not found: no chain'); return; }
      m.setValue(v.replace(FROM, TO));
      setTimeout(() => { document.querySelector('colab-run-button').shadowRoot.querySelector('#run-button').click(); window.__mqLog.push('clicked ' + new Date().toISOString()); }, 2000);
    }, 25000);
  }, 1000); return 'installed'; })()
```

セルが止まって 25 秒後もランタイムがつながっていれば（返されていれば「再接続」になる）、PLAN_FILE を書き換えて実行ボタンを押す。
次の計画の最初の run は `mq_fetch.py --require-tag done` なので、前の段が終わっていなければ始まらない。

注意：実行中のセルの実行ボタンは押さない（押すと止まる）。状態は `window.__mqLog`・`window.__mqDone` で読む。
