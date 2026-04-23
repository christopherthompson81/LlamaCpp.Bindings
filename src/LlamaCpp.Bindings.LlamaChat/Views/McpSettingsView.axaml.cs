using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class McpSettingsView : UserControl
{
    public McpSettingsView()
    {
        InitializeComponent();
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}
