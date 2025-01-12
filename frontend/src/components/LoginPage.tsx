//@ts-nocheck
import React, {useState, useEffect} from "react";
import { Box, Input, Button, VStack, Text } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { login } from '../api/auth';

function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const data = await login({ username, password });
      localStorage.setItem('accessToken', data.accessToken);
      navigate('/protected');
    } catch (err) {
      setError('Login failed.');
    }
  };

  return (
    <VStack spacing={4} p={6} boxShadow="lg" borderRadius="md" w={{ base: '90%', md: '400px' }}>
      <Text fontSize="xl">Login</Text>
      {error && <Text color="red.500">{error}</Text>}
      <Input
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />
      <Input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <Button colorScheme="blue" onClick={handleLogin}>Login</Button>
    </VStack>
  );
}

export default LoginPage;