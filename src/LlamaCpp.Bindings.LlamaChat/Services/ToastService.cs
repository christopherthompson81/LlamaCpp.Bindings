using System;
using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LlamaCpp.Bindings.LlamaChat.Services;

public enum ToastSeverity { Info, Success, Warning, Error }

public partial class Toast : ObservableObject
{
    public Guid Id { get; } = Guid.NewGuid();
    public string Title { get; init; } = "";
    public string Message { get; init; } = "";
    public ToastSeverity Severity { get; init; } = ToastSeverity.Info;
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;

    public bool IsInfo    => Severity == ToastSeverity.Info;
    public bool IsSuccess => Severity == ToastSeverity.Success;
    public bool IsWarning => Severity == ToastSeverity.Warning;
    public bool IsError   => Severity == ToastSeverity.Error;
}

/// <summary>
/// Process-wide toast sink. The main window binds a <c>ToastHost</c> to
/// <see cref="Current"/> — any code path can push a transient notification
/// here without plumbing the collection through every view model. Auto-
/// dismiss is driven by a single shared <see cref="DispatcherTimer"/>.
/// </summary>
public static class ToastService
{
    public static ObservableCollection<Toast> Current { get; } = new();

    public static TimeSpan DefaultDuration { get; } = TimeSpan.FromSeconds(4);

    public static void Show(string title, string? message, ToastSeverity severity = ToastSeverity.Info,
                             TimeSpan? duration = null)
    {
        var toast = new Toast
        {
            Title = title,
            Message = message ?? string.Empty,
            Severity = severity,
        };
        var ttl = duration ?? DefaultDuration;

        void Add()
        {
            Current.Add(toast);
            DispatcherTimer.RunOnce(() =>
            {
                Current.Remove(toast);
            }, ttl);
        }

        if (Dispatcher.UIThread.CheckAccess()) Add();
        else Dispatcher.UIThread.Post(Add);
    }

    public static void Info(string title, string? message = null) => Show(title, message, ToastSeverity.Info);
    public static void Success(string title, string? message = null) => Show(title, message, ToastSeverity.Success);
    public static void Warning(string title, string? message = null) => Show(title, message, ToastSeverity.Warning);
    public static void Error(string title, string? message = null) =>
        Show(title, message, ToastSeverity.Error, TimeSpan.FromSeconds(8));
}
