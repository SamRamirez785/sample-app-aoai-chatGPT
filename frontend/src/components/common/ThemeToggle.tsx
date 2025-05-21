import { IconButton } from '@fluentui/react'
import React, { useContext } from 'react'
import { ThemeContext } from '../../state/ThemeProvider'

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useContext(ThemeContext)
  const iconName = theme === 'light' ? 'ClearNight' : 'Sunny'

  return <IconButton ariaLabel="Toggle theme" iconProps={{ iconName }} onClick={toggleTheme} />
}

export default ThemeToggle
