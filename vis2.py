def visualize_performance(model_ft, test_loader, writer):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Test accuracy: {:.4f}%'.format(100 * correct / total))
    
    confusion_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(set(labels.numpy())))
    plt.xticks(tick_marks, sorted(set(labels.numpy())), rotation=45)
    plt.yticks(tick_marks, sorted(set(labels.numpy())))
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    classification_report = classification_report(labels.cpu().numpy(), predicted.cpu().numpy())
    with open('classification_report.txt', 'w') as f:
        f.write(str(classification_report))
    
    accuracy_score = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    with open('accuracy_score.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_score:.2f}")
