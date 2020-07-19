import torch

img = [i for i in range(784)]
img_tensor = torch.tensor(img, dtype = torch.float32)
img_tensor = img_tensor.reshape([1, 28, 28])

class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        self.max_val = float('-inf')
        def search_ball(nums, adds):
            if len(nums) <= 1:
                adds += nums[0]
                self.max_val = max(self.max_val, adds)
                return
            else:
                for idx, val in enumerate(nums):
                    if idx == 0:
                        p = adds + val*nums[idx + 1]
                        search_ball(nums[1:], p)
                    elif idx == len(nums) - 1:
                        p = adds + val*nums[idx - 1]
                        search_ball(nums[:len(nums)-1], p)
                    else:
                        p = adds + val*nums[idx-1]*nums[idx+1]
                        search_ball(nums[:idx]+nums[idx+1:], p)
        search_ball(nums, 0)
        return self.max_val

A = Solution()
print(A.maxCoins([7,9,8,0,1,2,3,4,5,1,1]))