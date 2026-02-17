const vscode = require("vscode");
const child_process = require("child_process");
const os = require("os");

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  console.log("CodeChat AI extension is now active!");

  let disposable = vscode.commands.registerCommand(
    "codechat.start",
    function () {
      // Create a new terminal
      const terminal = vscode.window.createTerminal("CodeChat AI");
      terminal.show();

      // Send the command to start the CLI
      // We assume 'codechat' is in the path. If not, we might need to look for the python script.
      // For better robustness, we can try to find the python interpreter and run the module.

      // Simple approach: assume 'codechat' command is installed via pip
      terminal.sendText("codechat --start");

      // Robust fallback approach if needed (can be added later):
      // terminal.sendText(`python "${context.extensionPath}/../cli.py" --start`);
    },
  );

  context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
