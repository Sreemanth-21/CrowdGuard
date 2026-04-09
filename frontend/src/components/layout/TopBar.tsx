interface TopBarProps {
  videoSource?: {
    type: 'webcam' | 'upload' | null;
    name: string;
  };
  isConnected?: boolean;
}

function TopBar({ videoSource, isConnected = false }: TopBarProps) {
  const getSourceDisplay = () => {
    if (!videoSource?.type) {
      return {
        text: 'NO SOURCE',
        showIndicator: false,
      };
    }

    if (videoSource.type === 'webcam') {
      return {
        text: 'WEBCAM LIVE',
        showIndicator: true,
      };
    }

    return {
      text: videoSource.name,
      showIndicator: true,
    };
  };

  const { text, showIndicator } = getSourceDisplay();

  return (
    <header className="fixed top-0 left-60 right-0 h-14 bg-navy-800 border-b border-navy-700 flex items-center px-6 z-10">
      <div className="flex items-center gap-3">
        {/* Source Name */}
        <span className="text-sm font-body font-semibold text-white">
          {text}
        </span>

        {/* Pulsing Indicator Dot */}
        {showIndicator && isConnected && (
          <div className="relative">
            <div className="w-2.5 h-2.5 bg-green-500 rounded-full"></div>
            <div className="absolute inset-0 w-2.5 h-2.5 bg-green-500 rounded-full animate-ping"></div>
          </div>
        )}
      </div>
    </header>
  );
}

export default TopBar;
