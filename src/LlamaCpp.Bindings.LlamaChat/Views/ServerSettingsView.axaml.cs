using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace LlamaCpp.Bindings.LlamaChat.Views;

public partial class ServerSettingsView : UserControl
{
    public ServerSettingsView()
    {
        InitializeComponent();
    }

    private void InitializeComponent() => AvaloniaXamlLoader.Load(this);
}
