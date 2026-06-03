import './index.css';

interface ConfirmModalProps {
  open: boolean;
  message: string[];
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmModal = ({
  open,
  message,
  onConfirm,
  onCancel,
}: ConfirmModalProps) => {
  if (!open) return null;

  return (
    <div className="confirm-overlay">
      <div className="confirm-modal">
        {message.map((string, index) => (
          <p key={index}>{string}</p>
        ))}
        <div className="buttons">
          <button onClick={onConfirm}>OK</button>
          <button onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmModal;
