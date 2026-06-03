import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ConfirmModal from '.';
import userEvent from '@testing-library/user-event';

describe('ConfirmModal', () => {
  let onConfirm: ReturnType<typeof vi.fn>;
  let onCancel: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onConfirm = vi.fn();
    onCancel = vi.fn();

    render(
      <ConfirmModal
        open={true}
        message={['this is one paragraph', 'another paragraph']}
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );
  });

  it('displays the text passed in props', () => {
    expect(screen.getByText('this is one paragraph')).toBeInTheDocument();

    expect(screen.getByText('another paragraph')).toBeInTheDocument();
  });

  it('calls onConfirm when OK is clicked', async () => {
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'OK' }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });
  it('calls onCancel when Cancel is clicked', async () => {
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
