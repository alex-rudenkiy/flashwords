//@ts-nocheck
import React, {useRef, useState} from 'react';
import {useQuery, useMutation, useQueryClient} from 'react-query';
import {
    Box,
    Button,
    Table,
    Spinner,
    Input,
    Textarea,
    Stack,
    PaginationPrevTrigger,
    HStack,
    PaginationRoot, PaginationNextTrigger
} from '@chakra-ui/react';
import {IconButton} from '@chakra-ui/react';
import axios from 'axios';
import {useToast} from '@chakra-ui/toast';
import {
    DrawerActionTrigger,
    DrawerBackdrop,
    DrawerBody,
    DrawerCloseTrigger,
    DrawerContent,
    DrawerFooter,
    DrawerHeader,
    DrawerRoot,
    DrawerTitle,
    DrawerTrigger
} from "@chakra-ui/react"

import {FiEdit2} from 'react-icons/fi';
import {AiFillDelete} from 'react-icons/ai';
import {getToken} from "../Setting";

export function VocabularyList() {
    const toast = useToast();
    const [isWordDialogDrawerOpen, setIsWordDialogDrawerOpen] = useState(false);
    const [isBatchDialogDrawerOpen, setIsBatchDialogDrawerOpen] = useState(false);


    const [editMode, setEditMode] = useState(false);
    const [editWordId, setEditWordId] = useState(null);
    const [newWord, setNewWord] = useState({foreignWord: '', nativeWord: '', description: ''});
    const [csvData, setCsvData] = useState('');

    const {
        data, isLoading, error
    } = useQuery(['vocabulary'], () => axios.get('/users/0/words', {headers: {Authorization: `Bearer ${getToken()}`}}).then((res) => res.data));
    const queryClient = useQueryClient();

    const deleteMutation = useMutation((wordId) => axios.delete(`/users/0/words/${wordId}`, {headers: {Authorization: `Bearer ${getToken()}`}}), {
        onSuccess: () => {
            queryClient.invalidateQueries(['vocabulary']);
            toast({title: 'Word deleted successfully.', status: 'success', duration: 3000, isClosable: true});
        }, onError: () => {
            toast({title: 'Failed to delete word.', status: 'error', duration: 3000, isClosable: true});
        },
    });

    const addOrUpdateMutation = useMutation((word) => {
        if (editMode) {
            return axios.put(`/users/0/words/${editWordId}`, word, {headers: {Authorization: `Bearer ${getToken()}`}});
        }
        return axios.post('/users/0/words', word, {headers: {Authorization: `Bearer ${getToken()}`}});
    }, {
        onSuccess: () => {
            queryClient.invalidateQueries(['vocabulary']);
            toast({
                title: `Word ${editMode ? 'updated' : 'added'} successfully.`,
                status: 'success',
                duration: 3000,
                isClosable: true
            });
            setIsWordDialogDrawerOpen(false);
            setNewWord({foreignWord: '', nativeWord: '', description: ''});
            setEditMode(false);
            setEditWordId(null);
        }, onError: () => {
            toast({title: 'Failed to save word.', status: 'error', duration: 3000, isClosable: true});
        },
    });

    const batchImportMutation = useMutation((csvData) => axios.post('/users/0/words/import', csvData, {headers: {Authorization: `Bearer ${getToken()}`}}), {
        onSuccess: () => {
            queryClient.invalidateQueries(['vocabulary']);
            toast({title: 'Words imported successfully.', status: 'success', duration: 3000, isClosable: true});
            setCsvData('');
        }, onError: () => {
            toast({title: 'Failed to import words.', status: 'error', duration: 3000, isClosable: true});
        },
    });

    if (isLoading) return <Spinner/>;
    if (error) return <Box>Error loading vocabulary</Box>;

    const handleDelete = (wordId) => {
        deleteMutation.mutate(wordId);
    };

    const handleSaveWord = () => {
        addOrUpdateMutation.mutate(newWord);
    };

    const handleBatchImport = () => {
        const words = csvData.split('\n').map((line) => {
            const [foreignWord, nativeWord, description] = line.split(';');
            console.log(line.split(';'));
            return {foreignWord, nativeWord, description};
        });
        batchImportMutation.mutate(words);
    };

    return (<>
            <Box mb={2} mt={6}>
                <Button mr={2} colorScheme="blue" onClick={() => setIsWordDialogDrawerOpen(true)}>
                    Add New Word
                </Button>
                <Button colorScheme="blue" onClick={() => setIsBatchDialogDrawerOpen(true)}>
                    Imports Words
                </Button>
            </Box>

            <Box>
                <DrawerRoot
                    placement='right'
                    open={isBatchDialogDrawerOpen} onClose={() => setIsBatchDialogDrawerOpen(false)}>
                    <DrawerBackdrop/>

                    <DrawerContent offset="4" rounded="md"
                                   style={{maxWidth: 'none', zIndex: 100000, marginBottom: '2em'}}>
                        <DrawerHeader>
                            <DrawerTitle>Import Words (CSV format)</DrawerTitle>
                            <DrawerCloseTrigger/>
                        </DrawerHeader>
                        <DrawerBody>
                            <Textarea
                                id="csvImport"
                                value={csvData}
                                onChange={(e) => setCsvData(e.target.value)}
                                placeholder="foreignWord,nativeWord,description"
                            />

                        </DrawerBody>
                        <DrawerFooter>
                            <Button colorScheme="blue" onClick={handleBatchImport}
                                    isLoading={batchImportMutation.isLoading}>
                                Import
                            </Button>
                            <Button onClick={() => setIsBatchDialogDrawerOpen(false)}>Cancel</Button>
                        </DrawerFooter>
                    </DrawerContent>
                </DrawerRoot>

                <DrawerRoot
                    open={isWordDialogDrawerOpen}
                    onClose={() => setIsWordDialogDrawerOpen(false)}
                >
                    <DrawerBackdrop/>


                    <DrawerContent offset="4" rounded="md"
                                   style={{maxWidth: 'none', zIndex: 100000, marginBottom: '2em'}}>
                        <DrawerHeader>
                            <DrawerTitle>{editMode ? 'Edit Word' : 'Add New Word'}</DrawerTitle>
                            <DrawerCloseTrigger/>
                        </DrawerHeader>
                        <DrawerBody>
                            <Box mb={3}>
                                <label htmlFor="foreignWord">Foreign Word</label>
                                <Input
                                    id="foreignWord"
                                    value={newWord.foreignWord}
                                    onChange={(e) => setNewWord({...newWord, foreignWord: e.target.value})}
                                />
                            </Box>
                            <Box mb={3}>
                                <label htmlFor="nativeWord">Native Word</label>
                                <Input
                                    id="nativeWord"
                                    value={newWord.nativeWord}
                                    onChange={(e) => setNewWord({...newWord, nativeWord: e.target.value})}
                                />
                            </Box>
                            <Box mb={3}>
                                <label htmlFor="description">Description</label>
                                <Input
                                    id="description"
                                    value={newWord.description}
                                    onChange={(e) => setNewWord({...newWord, description: e.target.value})}
                                />
                            </Box>
                        </DrawerBody>
                        <DrawerFooter>
                            <Button colorScheme="blue" mr={3} onClick={handleSaveWord}
                                    isLoading={addOrUpdateMutation.isLoading}>
                                Save
                            </Button>
                            <Button onClick={() => setIsWordDialogDrawerOpen(false)}>Cancel</Button>
                        </DrawerFooter>
                    </DrawerContent>
                </DrawerRoot>


                <Table.Root size="sm" interactive >
                    <Table.Header>
                        <Table.Row>
                            <Table.ColumnHeader>Foreign Word</Table.ColumnHeader>
                            <Table.ColumnHeader>Native Word</Table.ColumnHeader>
                            <Table.ColumnHeader hideBelow="md">Description</Table.ColumnHeader>
                            <Table.ColumnHeader>Actions</Table.ColumnHeader>
                        </Table.Row>
                    </Table.Header>
                    <Table.Body>
                        {data?.map((word) => (<Table.Row key={word.id}>
                                <Table.Cell>{word['foreign_word']}</Table.Cell>
                                <Table.Cell>{word['native_word']}</Table.Cell>
                                <Table.Cell hideBelow="md">{word.description}</Table.Cell>
                                <Table.Cell>
                                    <IconButton
                                        aria-label="Edit word"
                                        mr={2}
                                        onClick={() => {
                                            setEditMode(true);
                                            setEditWordId(word.id);
                                            setNewWord({
                                                foreignWord: word['foreign_word'],
                                                nativeWord: word['native_word'],
                                                description: word.description
                                            });
                                            setIsWordDialogDrawerOpen(true);
                                        }}
                                    >
                                        <FiEdit2 size={'tiny'}/>
                                    </IconButton>
                                    <IconButton
                                        aria-label="Delete word"
                                        colorScheme="red"
                                        onClick={() => handleDelete(word.id)}
                                    >
                                        <AiFillDelete size={12}/>
                                    </IconButton>
                                </Table.Cell>
                            </Table.Row>))}
                    </Table.Body>
                </Table.Root>


            </Box>

        </>);
}
